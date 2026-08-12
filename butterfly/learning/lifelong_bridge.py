from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from ..checkpoint import load_entry
from ..config import ROOT, load_json, save_json
from ..registry import get_seed_entry
from ..training.runtime import best_device
from ..upgrade import strict_baseline
from .curriculum_graph import (
    load_graph,
    material_bindings,
    node_strategy,
    record_assessment,
    record_attempt,
    summary as curriculum_summary,
)
from .dynamic_exam import dynamic_suite_id, evaluate_bank, generate_bank


LIFELONG_CONFIG_PATH = ROOT / "config" / "lifelong_learning.json"
DIAGNOSTIC_DIR = ROOT / "reports" / "lifelong"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_lifelong_config() -> dict[str, Any]:
    cfg = load_json(LIFELONG_CONFIG_PATH, {})
    if not isinstance(cfg, dict) or int(cfg.get("schema_version", 0)) < 2:
        raise RuntimeError("Lifelong Learning V2 config is not installed")
    return cfg


def _stable_seed(*parts: str) -> int:
    raw = ":".join(parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _legacy_binding(capability: str, family: str | None):
    if not family:
        return None, None
    for node_id, binding in material_bindings().items():
        if (
            binding.get("legacy_capability") == capability
            and binding.get("legacy_family") == family
        ):
            return node_id, dict(binding)
    return None, None


def bootstrap_fixed_mastery(seed: dict[str, Any] | None = None) -> None:
    seed = seed or get_seed_entry()
    if not seed:
        return
    metrics, _ = strict_baseline(seed)
    suite = str(metrics.get("suite_id") or "unknown")
    mapping = {
        "foundation.language": "language_component",
        "foundation.conversation": "conversation_component",
        "foundation.instructions": "instruction_format_component",
        "foundation.epistemic": "epistemic_dialogue_component",
    }
    for node_id, metric in mapping.items():
        if metric in metrics:
            record_assessment(
                node_id,
                float(metrics[metric]),
                exam_fingerprint=f"benchmark:{suite}:{metric}:{seed['version']}",
                exam_kind="fixed_regression_canary",
                replace_mastery=True,
            )


def _diagnostic_path(seed: dict[str, Any]) -> Path:
    return DIAGNOSTIC_DIR / f"brain-{seed['version']}-{dynamic_suite_id()}.json"


def strict_dynamic_diagnostic(seed: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    seed = seed or get_seed_entry()
    if not seed:
        raise RuntimeError("No ACTIVE/LAB seed available for lifelong diagnostic")
    bootstrap_fixed_mastery(seed)
    path = _diagnostic_path(seed)
    if path.exists() and not force:
        cached = load_json(path)
        if isinstance(cached, dict) and cached.get("dynamic_suite_id") == dynamic_suite_id():
            return cached

    cfg = load_lifelong_config()
    dyn_cfg = cfg.get("dynamic_exam") or {}
    count = int(dyn_cfg.get("diagnostic_cases", 4))
    bindings = material_bindings()
    model, _, tokenizer = load_entry(seed, device=best_device())
    assessed: set[str] = set()
    results: dict[str, Any] = {}

    while True:
        state, _ = load_graph()
        ready = [
            node_id for node_id in sorted(bindings)
            if node_id not in assessed
            and node_id in state.get("nodes", {})
            and state["nodes"][node_id].get("status") != "locked"
        ]
        if not ready:
            break
        for node_id in ready:
            binding = bindings[node_id]
            family = str(binding["dynamic_family"])
            bank = generate_bank(
                family,
                _stable_seed(seed["version"], dynamic_suite_id(), node_id, "diagnostic"),
                count=count,
                mode="diagnostic",
            )
            exam = evaluate_bank(model, tokenizer, bank)
            record_assessment(
                node_id,
                float(exam["score"]),
                exam_fingerprint=bank["fingerprint"],
                exam_kind="dynamic_diagnostic",
                replace_mastery=True,
            )
            results[node_id] = {
                "family": family,
                "score": float(exam["score"]),
                "semantic": float(exam["semantic"]),
                "fingerprint": bank["fingerprint"],
                "cases": len(bank.get("cases") or []),
            }
            assessed.add(node_id)

    payload = {
        "schema_version": 1,
        "created_at": _utcnow(),
        "seed_version": seed["version"],
        "dynamic_suite_id": dynamic_suite_id(),
        "results": results,
        "curriculum": curriculum_summary(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, payload)
    return payload


def enrich_legacy_lesson(lesson: dict[str, Any]) -> dict[str, Any]:
    result = dict(lesson)
    node_id, binding = _legacy_binding(str(lesson.get("capability")), lesson.get("focus_family"))
    if not node_id or not binding:
        return result
    strategy = node_strategy(node_id)
    if not strategy:
        result["plateau"] = True
        return result
    cfg = load_lifelong_config().get("dynamic_exam") or {}
    state, _ = load_graph()
    node_row = state.get("nodes", {}).get(node_id) or {}
    result.update({
        "curriculum_node": node_id,
        "node_attempts": len(node_row.get("attempts") or []),
        "dynamic_family": binding["dynamic_family"],
        "corpus_aliases": list(binding.get("corpus_aliases") or lesson.get("corpus_aliases") or []),
        "strategy": strategy,
        "strategy_id": str(strategy["id"]),
        "selection_min_delta": float(cfg.get("selection_min_delta", 0.02)),
        "transfer_min_delta": float(cfg.get("transfer_min_delta", 0.015)),
        "acceptance_min_delta": float(cfg.get("acceptance_min_delta", 0.02)),
        "acceptance_min_score": float(cfg.get("acceptance_min_score", 0.60)),
    })
    return result


def curriculum_lessons(
    attempted: set[str] | None = None,
    exclude_nodes: set[str] | None = None,
) -> list[dict[str, Any]]:
    attempted = attempted or set()
    exclude_nodes = exclude_nodes or set()
    strict_dynamic_diagnostic(force=False)
    state, seed_spec = load_graph()
    specs = {row["id"]: row for row in seed_spec.get("nodes", [])}
    bindings = material_bindings()
    cfg = load_lifelong_config()
    dyn_cfg = cfg.get("dynamic_exam") or {}
    target_default = float((cfg.get("curriculum") or {}).get("mastery_threshold", 0.82))
    rows = []
    for node_id, binding in bindings.items():
        if node_id in exclude_nodes:
            continue
        row = state.get("nodes", {}).get(node_id)
        spec = specs.get(node_id)
        if not row or not spec or row.get("status") in {"locked", "mastered", "plateau"}:
            continue
        strategy = node_strategy(node_id)
        if not strategy:
            continue
        strategy_id = str(strategy["id"])
        attempt_key = f"{node_id}:{strategy_id}"
        if attempt_key in attempted:
            continue
        mastery = float(row.get("mastery") or 0.0)
        target = float(spec.get("target_mastery", target_default))
        gap = max(0.0, target - mastery)
        if gap <= 0:
            continue
        rows.append({
            "capability": f"lifelong:{node_id}",
            "metric": "dynamic_mastery",
            "score": mastery,
            "target": target,
            "gap": gap,
            "priority": gap + 0.01 / (1 + len(row.get("attempts") or [])),
            "recipe": binding["recipe"],
            "implementation": "neural",
            "trainable": True,
            "critical_count": 0,
            "focus_family": binding["dynamic_family"],
            "focus_metric": None,
            "focus_reason": "curriculum_mastery",
            "corpus_aliases": list(binding.get("corpus_aliases") or []),
            "curriculum_node": node_id,
            "dynamic_family": binding["dynamic_family"],
            "strategy": strategy,
            "strategy_id": strategy_id,
            "selection_min_delta": float(dyn_cfg.get("selection_min_delta", 0.02)),
            "transfer_min_delta": float(dyn_cfg.get("transfer_min_delta", 0.015)),
            "acceptance_min_delta": float(dyn_cfg.get("acceptance_min_delta", 0.02)),
            "acceptance_min_score": float(dyn_cfg.get("acceptance_min_score", 0.60)),
        })
    rows.sort(key=lambda item: (-item["priority"], item["curriculum_node"]))
    return rows


def lesson_attempt_key(lesson: dict[str, Any]) -> str:
    node = lesson.get("curriculum_node")
    strategy = lesson.get("strategy_id")
    if node and strategy:
        return f"{node}:{strategy}"
    return f"{lesson.get('capability')}:{lesson.get('focus_family') or '*'}"


def focus_target_for_lesson(lesson: dict[str, Any]) -> dict[str, Any]:
    strategy = dict(lesson.get("strategy") or {})
    dynamic_cfg = load_lifelong_config().get("dynamic_exam") or {}
    return {
        "family": lesson.get("focus_family"),
        "study_metric": lesson.get("focus_metric"),
        "reason": lesson.get("focus_reason"),
        "corpus_aliases": lesson.get("corpus_aliases") or [],
        "lifelong_mode": bool(lesson.get("curriculum_node")),
        "curriculum_node": lesson.get("curriculum_node"),
        "dynamic_family": lesson.get("dynamic_family"),
        "strategy_id": lesson.get("strategy_id"),
        "focus_repeat_delta": int(strategy.get("focus_repeat_delta", 0)),
        "lr_scale": float(strategy.get("lr_scale", 1.0)),
        "selection_min_delta": float(lesson.get("selection_min_delta", 0.02)),
        "transfer_min_delta": float(lesson.get("transfer_min_delta", 0.015)),
        "acceptance_min_delta": float(lesson.get("acceptance_min_delta", 0.02)),
        "acceptance_min_score": float(lesson.get("acceptance_min_score", 0.60)),
        "selection_cases": int(dynamic_cfg.get("selection_cases", 12)),
        "transfer_cases": int(dynamic_cfg.get("transfer_cases", 12)),
        "acceptance_cases": int(dynamic_cfg.get("acceptance_cases", 16)),
    }


def record_block_outcome(lesson: dict[str, Any], block_record: dict[str, Any]) -> None:
    node_id = lesson.get("curriculum_node")
    strategy_id = lesson.get("strategy_id")
    if not node_id or not strategy_id:
        return
    acceptance = block_record.get("lifelong_acceptance") or {}
    delta = acceptance.get("delta")
    status = str(block_record.get("status") or "unknown")
    outcome = {
        "promoted": "accepted",
        "lab_accepted": "accepted",
        "rejected": "rejected",
    }.get(status, status)
    record_attempt(node_id, strategy_id=strategy_id, outcome=outcome, score_delta=delta)
    if status in {"promoted", "lab_accepted"} and acceptance.get("fingerprint"):
        record_assessment(
            node_id,
            float(acceptance.get("candidate_score", 0.0)),
            exam_fingerprint=str(acceptance["fingerprint"]),
            exam_kind="dynamic_acceptance",
            replace_mastery=True,
        )


def research_needed() -> bool:
    cfg = load_lifelong_config()
    if bool((cfg.get("research") or {}).get("enabled", False)):
        return False
    info = curriculum_summary()
    return bool(info.get("next_research_targets"))
