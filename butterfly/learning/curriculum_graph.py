from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import ROOT, load_json, save_json


SEED_PATH = ROOT / "config" / "curriculum_seed.json"
CONFIG_PATH = ROOT / "config" / "lifelong_learning.json"
STATE_PATH = ROOT / ".butterfly" / "curriculum_graph.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_settings() -> dict[str, Any]:
    cfg = load_json(CONFIG_PATH, {})
    value = cfg.get("curriculum") if isinstance(cfg, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("config/lifelong_learning.json has no curriculum object")
    return value


def load_seed() -> dict[str, Any]:
    seed = load_json(SEED_PATH)
    if not isinstance(seed, dict) or seed.get("schema_version") != 1:
        raise RuntimeError("Invalid config/curriculum_seed.json")
    validate_seed(seed)
    return seed


def seed_fingerprint(seed: dict[str, Any]) -> str:
    raw = json.dumps(seed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "curriculum-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def validate_seed(seed: dict[str, Any]) -> None:
    rows = list(seed.get("nodes") or [])
    ids = [str(row.get("id")) for row in rows]
    if not rows:
        raise RuntimeError("Curriculum seed has no nodes")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate curriculum node ids")
    known = set(ids)
    for row in rows:
        missing = set(row.get("prerequisites") or []) - known
        if missing:
            raise RuntimeError(f"Unknown prerequisite(s) for {row.get('id')}: {sorted(missing)}")

    graph = {row["id"]: list(row.get("prerequisites") or []) for row in rows}
    visiting, visited = set(), set()

    def visit(node_id: str):
        if node_id in visited:
            return
        if node_id in visiting:
            raise RuntimeError(f"Curriculum prerequisite cycle at {node_id}")
        visiting.add(node_id)
        for parent in graph[node_id]:
            visit(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)


def _new_node_state(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "mastery": None,
        "confidence": 0.0,
        "status": "locked" if spec.get("prerequisites") else "available",
        "material_status": "none",
        "last_assessed_at": None,
        "last_studied_at": None,
        "assessments": [],
        "attempts": [],
        "strategy_failures": {},
        "plateau": False,
    }


def bootstrap_graph() -> dict[str, Any]:
    seed = load_seed()
    fingerprint = seed_fingerprint(seed)
    existing = load_json(STATE_PATH, {})
    if not isinstance(existing, dict):
        existing = {}

    old_nodes = existing.get("nodes") if isinstance(existing.get("nodes"), dict) else {}
    state = {
        "schema_version": 1,
        "seed_fingerprint": fingerprint,
        "updated_at": _utcnow(),
        "nodes": {},
        "history": list(existing.get("history") or [])[-500:],
    }
    for spec in seed["nodes"]:
        current = old_nodes.get(spec["id"])
        state["nodes"][spec["id"]] = dict(current) if isinstance(current, dict) else _new_node_state(spec)

    full_cfg = load_json(CONFIG_PATH, {})
    for node_id, binding in (full_cfg.get("material_bindings") or {}).items():
        if node_id in state["nodes"] and binding.get("source") == "internal_verified":
            state["nodes"][node_id]["material_status"] = "internal_verified"

    _refresh_statuses(state, seed)
    save_json(STATE_PATH, state)
    return state


def load_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    seed = load_seed()
    state = load_json(STATE_PATH)
    if not isinstance(state, dict):
        state = bootstrap_graph()
    if state.get("seed_fingerprint") != seed_fingerprint(seed):
        state = bootstrap_graph()
    _refresh_statuses(state, seed)
    return state, seed


def _spec_map(seed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in seed["nodes"]}


def _mastery(state: dict[str, Any], node_id: str) -> float:
    raw = (state.get("nodes", {}).get(node_id) or {}).get("mastery")
    return float(raw) if raw is not None else 0.0


def _refresh_statuses(state: dict[str, Any], seed: dict[str, Any]) -> None:
    settings = load_settings()
    prereq_threshold = float(settings.get("prerequisite_threshold", 0.75))
    mastery_threshold = float(settings.get("mastery_threshold", 0.82))
    specs = _spec_map(seed)

    for node_id, spec in specs.items():
        row = state["nodes"].setdefault(node_id, _new_node_state(spec))
        if row.get("plateau"):
            row["status"] = "plateau"
            continue
        mastery = row.get("mastery")
        if mastery is not None and float(mastery) >= max(mastery_threshold, float(spec.get("target_mastery", mastery_threshold))):
            row["status"] = "mastered"
            continue
        prereqs = spec.get("prerequisites") or []
        row["status"] = (
            "available"
            if all(_mastery(state, parent) >= prereq_threshold for parent in prereqs)
            else "locked"
        )


def record_assessment(
    node_id: str,
    score: float,
    *,
    exam_fingerprint: str,
    exam_kind: str,
    replace_mastery: bool = False,
) -> dict[str, Any]:
    state, seed = load_graph()
    specs = _spec_map(seed)
    if node_id not in specs:
        raise KeyError(node_id)
    score = max(0.0, min(1.0, float(score)))
    row = state["nodes"][node_id]
    for previous in row.get("assessments") or []:
        if (
            previous.get("exam_fingerprint") == str(exam_fingerprint)
            and previous.get("exam_kind") == str(exam_kind)
        ):
            return row
    assessment = {
        "at": _utcnow(),
        "score": score,
        "exam_fingerprint": str(exam_fingerprint),
        "exam_kind": str(exam_kind),
    }
    assessments = list(row.get("assessments") or [])
    assessments.append(assessment)
    row["assessments"] = assessments[-50:]
    old = row.get("mastery")
    row["mastery"] = score if old is None or replace_mastery else (0.65 * float(old) + 0.35 * score)
    if replace_mastery:
        # A fresh current-brain exam supersedes stale mastery. Strategy plateau
        # is reopened only when the new brain materially changed this node, not
        # merely because some unrelated curriculum node produced a new LAB.
        target_mastery = float(specs[node_id].get("target_mastery", 0.82))
        materially_improved = (
            old is None
            or score >= float(old) + 0.05
            or score >= target_mastery
        )
        if materially_improved:
            row["plateau"] = False
            row["strategy_failures"] = {}
    row["confidence"] = min(1.0, float(row.get("confidence", 0.0)) + 0.12)
    row["last_assessed_at"] = assessment["at"]
    state["history"].append({"event": "assessment", "node": node_id, **assessment})
    state["history"] = state["history"][-500:]
    _refresh_statuses(state, seed)
    state["updated_at"] = _utcnow()
    save_json(STATE_PATH, state)
    return row


def record_attempt(
    node_id: str,
    *,
    strategy_id: str,
    outcome: str,
    score_delta: float | None = None,
) -> dict[str, Any]:
    state, seed = load_graph()
    specs = _spec_map(seed)
    if node_id not in specs:
        raise KeyError(node_id)
    settings = load_settings()
    row = state["nodes"][node_id]
    attempt = {
        "at": _utcnow(),
        "strategy_id": str(strategy_id),
        "outcome": str(outcome),
        "score_delta": None if score_delta is None else float(score_delta),
    }
    attempts = list(row.get("attempts") or [])
    attempts.append(attempt)
    row["attempts"] = attempts[-30:]
    row["last_studied_at"] = attempt["at"]

    failures = dict(row.get("strategy_failures") or {})
    if outcome in {"rejected", "no_progress", "regressed"}:
        failures[str(strategy_id)] = int(failures.get(str(strategy_id), 0)) + 1
    elif outcome in {"improved", "mastered", "accepted"}:
        failures[str(strategy_id)] = 0
    row["strategy_failures"] = failures

    failed_strategies = sum(1 for value in failures.values() if int(value) > 0)
    row["plateau"] = failed_strategies >= int(settings.get("plateau_after_failed_strategies", 3))
    state["history"].append({"event": "attempt", "node": node_id, **attempt})
    state["history"] = state["history"][-500:]
    _refresh_statuses(state, seed)
    state["updated_at"] = _utcnow()
    save_json(STATE_PATH, state)
    return row


def mark_material(node_id: str, status: str) -> None:
    state, seed = load_graph()
    if node_id not in state["nodes"]:
        raise KeyError(node_id)
    state["nodes"][node_id]["material_status"] = str(status)
    state["updated_at"] = _utcnow()
    save_json(STATE_PATH, state)


def choose_next(*, require_material: bool = False, limit: int = 10) -> list[dict[str, Any]]:
    state, seed = load_graph()
    settings = load_settings()
    specs = _spec_map(seed)
    scored = []

    stage_rank = {"foundation": 0, "core": 1, "intermediate": 2, "advanced": 3}
    for node_id, row in state["nodes"].items():
        if row.get("status") != "available":
            continue
        if require_material and row.get("material_status") not in {"verified_packet", "internal_verified"}:
            continue
        spec = specs[node_id]
        mastery = float(row["mastery"]) if row.get("mastery") is not None else 0.0
        weakness = 1.0 - mastery
        novelty = 1.0 / (1.0 + len(row.get("attempts") or []))
        prereqs = spec.get("prerequisites") or []
        prereq_score = (
            sum(_mastery(state, parent) for parent in prereqs) / len(prereqs)
            if prereqs else 1.0
        )
        score = (
            float(settings.get("weakness_weight", 0.55)) * weakness
            + float(settings.get("novelty_weight", 0.20)) * novelty
            + float(settings.get("prerequisite_weight", 0.25)) * prereq_score
            - 0.03 * stage_rank.get(spec.get("stage"), 4)
        )
        scored.append({
            "id": node_id,
            "title": spec["title"],
            "domain": spec["domain"],
            "stage": spec["stage"],
            "mastery": row.get("mastery"),
            "material_status": row.get("material_status"),
            "attempts": len(row.get("attempts") or []),
            "priority": score,
        })
    scored.sort(key=lambda item: (-item["priority"], item["stage"], item["id"]))
    return scored[:int(limit)]


def material_bindings() -> dict[str, dict[str, Any]]:
    cfg = load_json(CONFIG_PATH, {})
    value = cfg.get("material_bindings") if isinstance(cfg, dict) else None
    return dict(value or {})


def node_strategy(node_id: str) -> dict[str, Any] | None:
    state, seed = load_graph()
    if node_id not in state.get("nodes", {}):
        raise KeyError(node_id)
    row = state["nodes"][node_id]
    if row.get("plateau"):
        return None
    settings = load_settings()
    strategies = list(settings.get("study_strategies") or [])
    if not strategies:
        strategies = [{"id": "balanced", "focus_repeat_delta": 0, "lr_scale": 1.0}]
    failures = dict(row.get("strategy_failures") or {})
    ranked = sorted(
        strategies,
        key=lambda item: (int(failures.get(str(item.get("id")), 0)), strategies.index(item)),
    )
    return dict(ranked[0]) if ranked else None


def summary() -> dict[str, Any]:
    state, seed = load_graph()
    specs = _spec_map(seed)
    counts = {}
    domains = {}
    for node_id, row in state["nodes"].items():
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
        domain = specs[node_id]["domain"]
        domains[domain] = domains.get(domain, 0) + 1
    return {
        "seed_fingerprint": state["seed_fingerprint"],
        "nodes": len(state["nodes"]),
        "statuses": counts,
        "domains": domains,
        "next_research_targets": choose_next(require_material=False, limit=12),
        "next_trainable_with_verified_material": choose_next(require_material=True, limit=12),
    }


def print_status() -> None:
    info = summary()
    print("ButterflyAI Lifelong Curriculum Graph")
    print(f"Fingerprint : {info['seed_fingerprint']}")
    print(f"Nodes       : {info['nodes']}")
    print("Statuses    : " + ", ".join(f"{k}={v}" for k, v in sorted(info["statuses"].items())))
    print("Domains     : " + ", ".join(f"{k}={v}" for k, v in sorted(info["domains"].items())))
    print("")
    print("Next research / curriculum targets:")
    for item in info["next_research_targets"]:
        mastery = "unknown" if item["mastery"] is None else f"{float(item['mastery']):.3f}"
        print(
            f"  {item['id']:32s} mastery={mastery:>7s} "
            f"material={item['material_status']:>8s} priority={item['priority']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap_graph()
        print("Curriculum graph bootstrap: PASS")
    if args.status:
        if args.json:
            print(json.dumps(summary(), indent=2, ensure_ascii=False))
        else:
            print_status()
    if not args.bootstrap and not args.status:
        parser.error("use --bootstrap and/or --status")


if __name__ == "__main__":
    main()
