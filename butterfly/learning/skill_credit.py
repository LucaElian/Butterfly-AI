from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from typing import Any

import torch

from ..checkpoint import load_entry
from ..config import ROOT, load_json, save_json
from .dynamic_exam import (
    evaluate_bank,
    fresh_bank_excluding,
    normalize_surface,
    record_bank_use,
)
from .evaluator import behavior_benchmark


CONFIG_PATH = ROOT / "config" / "lifelong_learning.json"
LEDGER_PATH = ROOT / ".butterfly" / "skill_credit_ledger.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_skill_credit_config() -> dict[str, Any]:
    cfg = load_json(CONFIG_PATH, {})
    value = cfg.get("skill_credit") if isinstance(cfg, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("config/lifelong_learning.json has no skill_credit object")
    return value


def _stable_seed(*parts: str) -> int:
    raw = ":".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _component_metrics(metrics: dict[str, Any]) -> list[str]:
    return sorted(
        key for key, value in metrics.items()
        if key.endswith("_component") and isinstance(value, (int, float))
    )


def fixed_safety_check(
    trial: dict[str, Any],
    seed: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    max_overall = float(cfg.get("max_overall_regression", 0.005))
    trial_score = float(trial.get("score", 0.0))
    seed_score = float(seed.get("score", 0.0))
    if trial_score < seed_score - max_overall:
        failures.append(
            f"overall score {trial_score:.4f} < seed {seed_score:.4f} - {max_overall:.4f}"
        )

    if bool(cfg.get("require_no_new_critical", True)):
        seed_critical = set(seed.get("critical_failures") or [])
        trial_critical = set(trial.get("critical_failures") or [])
        new_critical = sorted(trial_critical - seed_critical)
        if new_critical:
            failures.append("new critical failures: " + ", ".join(new_critical))

    max_component = float(cfg.get("max_component_regression", 0.01))
    for name in _component_metrics(seed):
        if name not in trial:
            continue
        current = float(trial[name])
        baseline = float(seed[name])
        if current < baseline - max_component:
            failures.append(
                f"{name} {current:.4f} < seed {baseline:.4f} - {max_component:.4f}"
            )

    return not failures, failures


def dynamic_credit_gate(
    current_score: float,
    seed_score: float,
    *,
    minimum_delta: float,
    minimum_score: float,
) -> tuple[bool, list[str]]:
    failures = []
    delta = float(current_score) - float(seed_score)
    if delta < float(minimum_delta):
        failures.append(
            f"skill delta {delta:+.4f} < required {float(minimum_delta):+.4f}"
        )
    if float(current_score) < float(minimum_score):
        failures.append(
            f"skill score {float(current_score):.4f} < floor {float(minimum_score):.4f}"
        )
    return not failures, failures


def _eligible_target(candidate_entry: dict[str, Any]) -> tuple[dict[str, Any], str, str] | None:
    metadata = candidate_entry.get("metadata") or {}
    target = metadata.get("focus_target") or {}
    if not bool(target.get("lifelong_mode")):
        return None
    node_id = target.get("curriculum_node")
    family = target.get("dynamic_family")
    if not node_id or not family:
        return None
    return dict(target), str(node_id), str(family)


def _validated_alpha_ladder(cfg: dict[str, Any]) -> list[float]:
    values = []
    for raw in cfg.get("alpha_ladder") or []:
        value = float(raw)
        if not (0.0 < value <= 1.0):
            raise RuntimeError("skill_credit alpha values must be in (0, 1]")
        values.append(value)
    values = sorted(set(values), reverse=True)
    if not values:
        raise RuntimeError("skill_credit alpha_ladder is empty")
    return values


def _interpolated_model(seed_model, candidate_model, alpha: float):
    trial = copy.deepcopy(seed_model)
    candidate_params = dict(candidate_model.named_parameters())
    trial_params = dict(trial.named_parameters())
    if set(candidate_params) != set(trial_params):
        raise RuntimeError("Candidate/seed parameter topology mismatch during skill credit")
    with torch.no_grad():
        for name, param in trial_params.items():
            candidate_param = candidate_params[name].detach().to(
                device=param.device, dtype=param.dtype
            )
            param.lerp_(candidate_param, float(alpha))
    return trial


def attempt_skill_credit_salvage(
    *,
    seed_entry: dict[str, Any],
    candidate_entry: dict[str, Any],
    candidate_model,
    tokenizer,
    seed_baseline: dict[str, Any],
    recipe: dict[str, Any],
) -> tuple[dict[str, Any] | None, Any | None]:
    cfg = load_skill_credit_config()
    if not bool(cfg.get("enabled", True)):
        return None, None

    eligible = _eligible_target(candidate_entry)
    if not eligible:
        return None, None
    focus_target, node_id, family = eligible

    experiment_id = str((candidate_entry.get("metadata") or {}).get("experiment_id") or "")
    target_version = str(candidate_entry.get("version") or "")
    if not experiment_id or not target_version:
        raise RuntimeError("Skill credit requires experiment and target lineage")

    seed_model, _, seed_tokenizer = load_entry(
        seed_entry, device=next(candidate_model.parameters()).device
    )
    if seed_tokenizer.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("Skill credit seed/candidate tokenizer mismatch")

    search_count = int(cfg.get("search_cases", 12))
    search_seed = _stable_seed(
        experiment_id, target_version, node_id, "skill-credit-search"
    )
    search_bank = fresh_bank_excluding(
        family,
        search_seed,
        count=search_count,
        mode="credit_search",
        forbidden_surfaces=set(),
    )
    record_bank_use(
        search_bank,
        purpose="skill_credit_search",
        experiment_id=experiment_id,
    )
    seed_search = evaluate_bank(seed_model, tokenizer, search_bank)
    seed_search_score = float(seed_search["score"])

    search_min_delta = float(cfg.get("search_min_delta", 0.015))
    minimum_score = float(cfg.get("minimum_skill_score", 0.60))
    chosen_model = None
    chosen_alpha = None
    chosen_fixed = None
    search_attempts = []

    print("")
    print("=== SUBJECT / SKILL CREDIT SALVAGE ===")
    print(f"node   : {node_id}")
    print(f"family : {family}")
    print(f"seed skill score on credit-search bank: {seed_search_score:.4f}")
    print("The full candidate failed LAB/ACTIVE; trying safe partial consolidation.")

    for alpha in _validated_alpha_ladder(cfg):
        trial = _interpolated_model(seed_model, candidate_model, alpha)
        trial_search = evaluate_bank(trial, tokenizer, search_bank)
        trial_search_score = float(trial_search["score"])
        skill_ok, skill_failures = dynamic_credit_gate(
            trial_search_score,
            seed_search_score,
            minimum_delta=search_min_delta,
            minimum_score=minimum_score,
        )
        row = {
            "alpha": float(alpha),
            "seed_skill_score": seed_search_score,
            "skill_score": trial_search_score,
            "skill_delta": trial_search_score - seed_search_score,
            "skill_gate_passed": bool(skill_ok),
            "skill_failures": list(skill_failures),
        }
        if not skill_ok:
            search_attempts.append(row)
            print(
                f"alpha={alpha:.3f} skill={trial_search_score:.4f} "
                f"delta={trial_search_score-seed_search_score:+.4f} -> SKILL FAIL"
            )
            del trial
            continue

        fixed_metrics = behavior_benchmark(trial, tokenizer)
        fixed_ok, fixed_failures = fixed_safety_check(
            fixed_metrics, seed_baseline, cfg
        )
        row.update({
            "fixed_score": float(fixed_metrics.get("score", 0.0)),
            "fixed_gate_passed": bool(fixed_ok),
            "fixed_failures": list(fixed_failures),
        })
        search_attempts.append(row)
        print(
            f"alpha={alpha:.3f} skill={trial_search_score:.4f} "
            f"delta={trial_search_score-seed_search_score:+.4f} "
            f"fixed={float(fixed_metrics.get('score',0)):.4f} "
            f"-> {'SAFE' if fixed_ok else 'FIXED FAIL'}"
        )
        if not fixed_ok:
            del trial
            continue

        chosen_model = trial
        chosen_alpha = float(alpha)
        chosen_fixed = fixed_metrics
        break

    if chosen_model is None:
        print("No alpha preserved the studied skill while keeping fixed capabilities safe.")
        return {
            "schema_version": 1,
            "passed": False,
            "node": node_id,
            "family": family,
            "reason": "no_safe_interpolation",
            "search_bank": search_bank["fingerprint"],
            "search_attempts": search_attempts,
        }, None

    forbidden = {
        normalize_surface(row["prompt"])
        for row in search_bank.get("cases", [])
    }
    acceptance_count = int(cfg.get("acceptance_cases", 16))
    acceptance_seed = _stable_seed(
        experiment_id, target_version, node_id, "skill-credit-acceptance"
    )
    acceptance_bank = fresh_bank_excluding(
        family,
        acceptance_seed,
        count=acceptance_count,
        mode="credit_acceptance",
        forbidden_surfaces=forbidden,
    )
    record_bank_use(
        acceptance_bank,
        purpose="skill_credit_acceptance",
        experiment_id=experiment_id,
    )
    seed_acceptance = evaluate_bank(seed_model, tokenizer, acceptance_bank)
    credit_acceptance = evaluate_bank(chosen_model, tokenizer, acceptance_bank)
    base_score = float(seed_acceptance["score"])
    current_score = float(credit_acceptance["score"])
    acceptance_min_delta = float(cfg.get("acceptance_min_delta", 0.02))
    acceptance_ok, acceptance_failures = dynamic_credit_gate(
        current_score,
        base_score,
        minimum_delta=acceptance_min_delta,
        minimum_score=minimum_score,
    )

    acceptance = {
        "curriculum_node": node_id,
        "family": family,
        "fingerprint": acceptance_bank["fingerprint"],
        "seed_score": base_score,
        "candidate_score": current_score,
        "delta": current_score - base_score,
        "required_delta": acceptance_min_delta,
        "required_score": minimum_score,
        "passed": bool(acceptance_ok),
    }

    print("")
    print("=== FRESH SUBJECT CREDIT ACCEPTANCE EXAM ===")
    print(f"alpha  : {chosen_alpha:.3f}")
    print(f"bank   : {acceptance_bank['fingerprint']}")
    print(
        f"score  : {base_score:.4f} -> {current_score:.4f} "
        f"delta={current_score-base_score:+.4f}"
    )
    print(
        f"gate   : delta>={acceptance_min_delta:+.4f} and "
        f"score>={minimum_score:.4f} -> {'PASS' if acceptance_ok else 'FAIL'}"
    )

    result = {
        "schema_version": 1,
        "passed": bool(acceptance_ok),
        "node": node_id,
        "family": family,
        "alpha": chosen_alpha,
        "seed_version": str(seed_entry.get("version")),
        "target_version": target_version,
        "experiment_id": experiment_id,
        "search_bank": search_bank["fingerprint"],
        "search_attempts": search_attempts,
        "acceptance": acceptance,
        "fixed_metrics": chosen_fixed,
        "fixed_score": float((chosen_fixed or {}).get("score", 0.0)),
        "focus_target": focus_target,
        "acceptance_failures": acceptance_failures,
    }
    if not acceptance_ok:
        del chosen_model
        return result, None
    return result, chosen_model


def _load_ledger() -> dict[str, Any]:
    value = load_json(LEDGER_PATH, {"schema_version": 1, "credits": []})
    if not isinstance(value, dict):
        value = {"schema_version": 1, "credits": []}
    value.setdefault("schema_version", 1)
    value.setdefault("credits", [])
    return value


def record_skill_credit(
    credit: dict[str, Any],
    *,
    result_version: str,
    benchmark_path: str | None,
    mode: str = "partial_consolidation",
) -> None:
    if not credit or not credit.get("passed"):
        return
    cfg = load_skill_credit_config()
    ledger = _load_ledger()
    row = {
        "at": _utcnow(),
        "node": credit.get("node"),
        "family": credit.get("family"),
        "mode": str(mode),
        "alpha": credit.get("alpha", 1.0),
        "seed_version": credit.get("seed_version"),
        "result_version": str(result_version),
        "experiment_id": credit.get("experiment_id"),
        "skill_score": (credit.get("acceptance") or {}).get("candidate_score"),
        "skill_delta": (credit.get("acceptance") or {}).get("delta"),
        "fixed_score": credit.get("fixed_score"),
        "benchmark": benchmark_path,
        "acceptance_fingerprint": (credit.get("acceptance") or {}).get("fingerprint"),
    }
    rows = list(ledger.get("credits") or [])
    rows.append(row)
    max_entries = max(1, int(cfg.get("ledger_max_entries", 500)))
    ledger["credits"] = rows[-max_entries:]
    save_json(LEDGER_PATH, ledger)


def record_full_candidate_credit(
    candidate_entry: dict[str, Any],
    *,
    result_version: str,
    fixed_score: float,
    benchmark_path: str | None,
) -> None:
    metadata = candidate_entry.get("metadata") or {}
    target = metadata.get("focus_target") or {}
    acceptance = metadata.get("lifelong_acceptance") or {}
    if not (
        target.get("lifelong_mode")
        and target.get("curriculum_node")
        and target.get("dynamic_family")
        and acceptance.get("passed")
    ):
        return
    credit = {
        "passed": True,
        "node": target["curriculum_node"],
        "family": target["dynamic_family"],
        "alpha": 1.0,
        "seed_version": metadata.get("seed_brain"),
        "experiment_id": metadata.get("experiment_id"),
        "fixed_score": float(fixed_score),
        "acceptance": acceptance,
    }
    record_skill_credit(
        credit,
        result_version=result_version,
        benchmark_path=benchmark_path,
        mode="full_candidate",
    )


def status() -> dict[str, Any]:
    ledger = _load_ledger()
    rows = list(ledger.get("credits") or [])
    by_node: dict[str, int] = {}
    for row in rows:
        node = str(row.get("node") or "unknown")
        by_node[node] = by_node.get(node, 0) + 1
    return {
        "credits": len(rows),
        "nodes_with_credit": len(by_node),
        "by_node": dict(sorted(by_node.items())),
        "recent": rows[-10:],
    }


def print_status() -> None:
    info = status()
    print("ButterflyAI Skill Credit Ledger")
    print(f"Credits          : {info['credits']}")
    print(f"Nodes with credit: {info['nodes_with_credit']}")
    if info["by_node"]:
        print("")
        print("Credits by curriculum node:")
        for node, count in info["by_node"].items():
            print(f"  {node:32s} {count}")


def self_test() -> None:
    cfg = {
        "max_overall_regression": 0.005,
        "max_component_regression": 0.01,
        "require_no_new_critical": True,
    }
    seed = {
        "score": 0.80,
        "critical_failures": ["old"],
        "a_component": 0.70,
        "b_component": 0.90,
    }
    safe = {
        "score": 0.798,
        "critical_failures": ["old"],
        "a_component": 0.72,
        "b_component": 0.895,
    }
    unsafe = {
        "score": 0.79,
        "critical_failures": ["old", "new"],
        "a_component": 0.72,
        "b_component": 0.87,
    }
    assert fixed_safety_check(safe, seed, cfg)[0]
    assert not fixed_safety_check(unsafe, seed, cfg)[0]
    assert dynamic_credit_gate(
        0.72, 0.68, minimum_delta=0.02, minimum_score=0.60
    )[0]
    assert not dynamic_credit_gate(
        0.69, 0.68, minimum_delta=0.02, minimum_score=0.60
    )[0]
    assert _validated_alpha_ladder(
        {"alpha_ladder": [0.25, 1.0, 0.5, 0.5]}
    ) == [1.0, 0.5, 0.25]
    print("Skill Credit self-test: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.status:
        print_status()
    if not args.status and not args.self_test:
        parser.error("use --status and/or --self-test")


if __name__ == "__main__":
    main()
