from __future__ import annotations

import json
import time

from .checkpoint import load_entry, move_model_artifacts, delete_model_artifacts
from .config import ROOT, MODELS_DIR, BENCHMARKS_DIR, load_promotion_policy, project_relpath
from .learning.evaluator import BENCHMARK_SUITE_ID, PROMOTION_THRESHOLDS, behavior_benchmark, print_benchmark, save_benchmark
from .registry import (
    append_history,
    compact_physical_models,
    get_active_entry,
    get_candidate_entry,
    get_entry,
    get_lab_entry,
    promote_to_active,
    promote_to_lab,
    remove_entry,
    update_entry,
)
from .training.runtime import best_device


def _suite_artifact_path(prefix: str):
    return BENCHMARKS_DIR / f"{prefix}-{BENCHMARK_SUITE_ID}.json"


def _baseline_path(entry):
    return _suite_artifact_path(f"baseline-brain-{entry['version']}")


def _comparison_path(active, seed, candidate):
    return _suite_artifact_path(
        f"comparison-active-{active['version']}-seed-{seed['version']}-candidate-{candidate['version']}"
    )


def strict_baseline(entry, force: bool = False):
    path = _baseline_path(entry)
    if path.exists() and not force:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("suite_id") == BENCHMARK_SUITE_ID:
            return value, path
    model, _, tokenizer = load_entry(entry, device=best_device())
    metrics = behavior_benchmark(model, tokenizer)
    save_benchmark(path, metrics)
    return metrics, path


def prepare_target(target_version: str, seed_version: str | None = None):
    seed = get_entry(seed_version) if seed_version else (get_lab_entry() or get_active_entry())
    active = get_active_entry()
    if not active:
        raise RuntimeError("No ACTIVE Butterfly brain.")
    if not seed:
        raise RuntimeError("No seed brain.")
    current_lab = get_lab_entry()
    if target_version in {active.get("version"), (current_lab or {}).get("version")}:
        raise RuntimeError("Target version must differ from ACTIVE and LAB.")
    if get_candidate_entry():
        raise RuntimeError("A candidate is already registered. Finish it before preparing another.")

    model, _, tokenizer = load_entry(seed, device="cpu")
    tokenizer_rel = (seed.get("metadata") or {}).get("tokenizer_path")
    if not tokenizer_rel:
        raise RuntimeError("Seed brain has no tokenizer lineage.")
    if model.cfg.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("Seed model/tokenizer vocabulary mismatch.")

    baseline, baseline_path = strict_baseline(seed)
    print("ButterflyAI generic experiment preflight OK.")
    print(f"Target brain           : {target_version}")
    print(f"Seed brain             : {seed['version']} ({seed.get('status')})")
    print(f"ACTIVE brain           : {active['version']} (READ ONLY until global promotion)")
    print(f"Parameters             : {model.parameter_count():,}")
    print(f"Tokenizer vocab        : {tokenizer.vocab_size:,}")
    print(f"Evaluator suite        : {BENCHMARK_SUITE_ID}")
    print(f"Seed baseline score    : {float(baseline.get('score', 0.0)):.4f}")
    print(f"Baseline file          : {baseline_path}")
    print("No weights, tokenizer, memory or corpus were modified.")
    return {
        "target_version": target_version,
        "seed_version": seed["version"],
        "suite_id": BENCHMARK_SUITE_ID,
        "baseline_score": baseline.get("score"),
        "baseline_path": str(baseline_path),
    }


def _no_regression(candidate, baseline, metrics: list[str], tolerance: float):
    failures = []
    for key in metrics:
        cand = float(candidate.get(key, 0.0))
        base = float(baseline.get(key, 0.0))
        if cand < base - tolerance:
            failures.append(f"{key}: {cand:.4f} < {base:.4f} - {tolerance:.4f}")
    return not failures, failures


def _critical_failures_for_categories(metrics, categories):
    failures = set(metrics.get("critical_failures") or [])
    if not categories:
        return failures
    categories = set(categories)
    id_to_category = {
        row.get("id"): row.get("category")
        for row in metrics.get("cases", [])
        if row.get("id")
    }
    return {
        case_id for case_id in failures
        if id_to_category.get(case_id) in categories
    }


def _lab_focus_check(candidate, seed_baseline, recipe, policy):
    focus = list(recipe.get("focus_metrics") or [])
    if not focus:
        return False, ["recipe has no focus_metrics"]

    lab_policy = policy.get("lab") or {}
    minimum = float(lab_policy.get("min_focus_delta", 0.05))
    require_all = bool(lab_policy.get("require_all_focus_metrics", True))
    deltas = {
        key: float(candidate.get(key, 0.0)) - float(seed_baseline.get(key, 0.0))
        for key in focus
    }

    if require_all:
        standard_focus_ok = all(delta >= minimum for delta in deltas.values())
    else:
        standard_focus_ok = sum(deltas.values()) / len(deltas) >= minimum

    acceptance = recipe.get("lab_acceptance") or {}
    focus_ok = standard_focus_ok
    accepted_by = "delta" if focus_ok else None

    if not focus_ok and bool(acceptance.get("allow_hard_gate_closure", False)):
        checks = []
        for key in focus:
            threshold = PROMOTION_THRESHOLDS.get(key)
            if threshold is None:
                checks.append(False)
                continue
            base = float(seed_baseline.get(key, 0.0))
            cur = float(candidate.get(key, 0.0))
            checks.append(base < float(threshold) <= cur)
        gate_closure_ok = all(checks) if require_all else any(checks)
        if gate_closure_ok:
            focus_ok = True
            accepted_by = "hard_gate_closure"

    if not focus_ok and bool(acceptance.get("allow_critical_repair", False)):
        categories = list(acceptance.get("critical_categories") or [])
        seed_category_failures = _critical_failures_for_categories(seed_baseline, categories)
        cand_category_failures = _critical_failures_for_categories(candidate, categories)

        seed_all = set(seed_baseline.get("critical_failures") or [])
        cand_all = set(candidate.get("critical_failures") or [])
        no_new_critical = cand_all.issubset(seed_all)
        repaired = (
            bool(seed_category_failures)
            and cand_category_failures.issubset(seed_category_failures)
            and len(cand_category_failures) < len(seed_category_failures)
        )

        max_focus_regression = float(acceptance.get("max_focus_regression", 0.01))
        focus_nonregression = all(
            float(candidate.get(key, 0.0))
            >= float(seed_baseline.get(key, 0.0)) - max_focus_regression
            for key in focus
        )
        focus_above_hard_floor = all(
            PROMOTION_THRESHOLDS.get(key) is None
            or float(candidate.get(key, 0.0)) >= float(PROMOTION_THRESHOLDS[key])
            for key in focus
        )

        if repaired and no_new_critical and focus_nonregression and focus_above_hard_floor:
            focus_ok = True
            accepted_by = "critical_repair"

    failures = []
    if not focus_ok:
        failures.append(
            "focus delta: "
            + ", ".join(f"{key}={delta:+.4f}" for key, delta in deltas.items())
            + f" (required {minimum:+.4f}; no configured alternate LAB acceptance passed)"
        )

    protected = list(recipe.get("protected_metrics") or [])
    tolerance = float(lab_policy.get("max_protected_regression", 0.03))
    protected_ok, protected_failures = _no_regression(
        candidate, seed_baseline, protected, tolerance
    )
    failures.extend(protected_failures)

    if focus_ok and accepted_by and accepted_by != "delta":
        # This is kept in the comparison artifact via recipe/policy behavior,
        # while normal successful runs remain quiet.
        pass

    return focus_ok and protected_ok, failures


def _active_check(candidate, active_baseline, recipe, policy):
    active_policy = policy.get("active") or {}
    minimum = float(active_policy.get("min_overall_delta", 0.03))
    improvement = float(candidate.get("score", 0.0)) - float(active_baseline.get("score", 0.0))
    gates_ok = bool(candidate.get("promotion_eligible"))
    if not bool(active_policy.get("requires_all_hard_gates", True)):
        gates_ok = True

    protected = sorted(set(
        list(recipe.get("protected_metrics") or [])
        + list(recipe.get("focus_metrics") or [])
    ))
    tolerance = float(active_policy.get("max_protected_regression", 0.02))
    regress_ok, regressions = _no_regression(candidate, active_baseline, protected, tolerance)

    failures = []
    if improvement < minimum:
        failures.append(f"overall improvement {improvement:+.4f} < {minimum:+.4f}")
    if not gates_ok:
        failures.extend(candidate.get("promotion_blockers", []))
    failures.extend(regressions)
    return improvement >= minimum and gates_ok and regress_ok, failures


def evaluate_candidate(target_version: str | None, recipe: dict):
    candidate = get_candidate_entry()
    if not candidate:
        raise RuntimeError("No candidate brain is registered.")
    if target_version and candidate.get("version") != target_version:
        raise RuntimeError("Registered candidate does not match current experiment target.")

    active = get_active_entry()
    if not active:
        raise RuntimeError("No ACTIVE brain.")
    seed_version = (candidate.get("metadata") or {}).get("seed_brain")
    seed = get_entry(seed_version)
    if not seed:
        raise RuntimeError("Candidate seed lineage is no longer registered.")

    active_baseline, active_path = strict_baseline(active)
    if seed["version"] == active["version"]:
        seed_baseline, seed_path = active_baseline, active_path
    else:
        seed_baseline, seed_path = strict_baseline(seed)

    print(f"\nEvaluating candidate {candidate['version']} with suite {BENCHMARK_SUITE_ID}...")
    model, _, tokenizer = load_entry(candidate, device=best_device())
    metrics = behavior_benchmark(model, tokenizer)
    print_benchmark(metrics)

    policy = load_promotion_policy()
    active_ok, active_failures = _active_check(metrics, active_baseline, recipe, policy)
    lab_ok, lab_failures = _lab_focus_check(metrics, seed_baseline, recipe, policy)

    report = {
        "suite_id": BENCHMARK_SUITE_ID,
        "active_version": active["version"],
        "seed_version": seed["version"],
        "candidate_version": candidate["version"],
        "active_baseline": active_baseline,
        "seed_baseline": seed_baseline,
        "candidate": metrics,
        "active_promotion": {"eligible": active_ok, "failures": active_failures},
        "lab_acceptance": {"eligible": lab_ok, "failures": lab_failures},
        "created_at": time.time(),
    }
    report_path = _comparison_path(active, seed, candidate)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    candidate_path = MODELS_DIR / candidate["path"]
    target = candidate["version"]

    if active_ok:
        canonical = MODELS_DIR / f"butterfly-v{target}.safetensors"
        if candidate_path.resolve() != canonical.resolve():
            move_model_artifacts(candidate_path, canonical)
            update_entry(target, path=canonical.name)
        update_entry(target, score=metrics["score"], metadata={
            "benchmark": project_relpath(report_path),
            "suite_id": BENCHMARK_SUITE_ID,
            "storage_format": "safetensors-weights-only",
            "optimizer_included": False,
        })
        promote_to_active(target)
        append_history(target, "promoted", score=metrics["score"], metadata={
            "benchmark": project_relpath(report_path),
            "suite_id": BENCHMARK_SUITE_ID,
            "seed_brain": seed["version"],
        })
        compact_physical_models()
        print(f"\nPROMOTED to ACTIVE: {target}")
        return "PROMOTED", report_path, metrics

    if lab_ok:
        canonical = MODELS_DIR / f"butterfly-v{target}-lab.safetensors"
        if candidate_path.resolve() != canonical.resolve():
            move_model_artifacts(candidate_path, canonical)
            update_entry(target, path=canonical.name)
        update_entry(target, score=metrics["score"], metadata={
            "benchmark": project_relpath(report_path),
            "suite_id": BENCHMARK_SUITE_ID,
            "storage_format": "safetensors-weights-only",
            "optimizer_included": False,
        })
        promote_to_lab(target)
        append_history(target, "lab_accepted", score=metrics["score"], metadata={
            "benchmark": project_relpath(report_path),
            "suite_id": BENCHMARK_SUITE_ID,
            "seed_brain": seed["version"],
            "focus_metrics": recipe.get("focus_metrics", []),
        })
        compact_physical_models()
        print(f"\nACCEPTED as LAB: {target}")
        print("ACTIVE was not replaced because the candidate did not pass global promotion.")
        return "LAB_ACCEPTED", report_path, metrics

    append_history(target, "rejected", score=metrics.get("score"), metadata={
        "benchmark": project_relpath(report_path),
        "suite_id": BENCHMARK_SUITE_ID,
        "seed_brain": seed["version"],
        "lab_failures": lab_failures,
        "active_failures": active_failures,
    })
    delete_model_artifacts(candidate_path)
    remove_entry(target)
    print("\nREJECTED: candidate did not qualify for LAB or ACTIVE.")
    print("LAB blockers:")
    for reason in lab_failures:
        print("  -", reason)
    print("Candidate physical weights deleted; history and benchmark kept.")
    return "REJECTED", report_path, metrics
