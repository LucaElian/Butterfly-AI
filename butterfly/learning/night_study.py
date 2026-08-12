from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from ..config import ROOT, load_json, save_json
from ..experiments import (
    clear_terminal_experiment,
    create_experiment,
    load_current_experiment,
)
from ..memory import MemoryStore
from ..registry import get_seed_entry
from ..upgrade import strict_baseline
from .evaluator import PROMOTION_THRESHOLDS
from .lifelong_bridge import (
    curriculum_lessons, enrich_legacy_lesson, focus_target_for_lesson,
    lesson_attempt_key, record_block_outcome, research_needed, strict_dynamic_diagnostic,
)


CONFIG_PATH = ROOT / "config" / "night_study.json"
LATEST_PLAN_PATH = ROOT / "reports" / "night-study-plan.json"
LATEST_REPORT_PATH = ROOT / "reports" / "night-study-latest.json"
HISTORY_PATH = ROOT / ".butterfly" / "night_study_history.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_night_config() -> dict[str, Any]:
    cfg = load_json(CONFIG_PATH)
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise RuntimeError("Invalid config/night_study.json")
    return cfg


def _capability_registry() -> dict[str, Any]:
    from ..runtime import load_capabilities
    return load_capabilities().get("capabilities", {})


def _verified_experience_count() -> int:
    # approved_experiences already enforces verified + quality + unused.
    return len(MemoryStore().approved_experiences(limit=5000))


def _critical_failures_by_category(metrics: dict[str, Any]) -> dict[str, int]:
    failed = set(metrics.get("critical_failures") or [])
    counts: dict[str, int] = {}
    for row in metrics.get("cases", []):
        if row.get("id") not in failed:
            continue
        category = str(row.get("category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return counts


def _critical_cases(metrics: dict[str, Any], category: str | None) -> list[dict[str, Any]]:
    if not category:
        return []

    # Benchmark result rows intentionally keep evaluation outputs compact and do
    # not need to duplicate every canonical case field. Failure-driven planning
    # does need validator/skill metadata, so enrich failed rows from the canonical
    # held-out case definitions by stable case id.
    from .evaluator import CASES

    specs = {case.get("id"): case for case in CASES if case.get("id")}
    results = {
        row.get("id"): row
        for row in metrics.get("cases", [])
        if row.get("id")
    }

    out: list[dict[str, Any]] = []
    for case_id in metrics.get("critical_failures") or []:
        merged = dict(specs.get(case_id) or {})
        merged.update(results.get(case_id) or {})
        if merged.get("category") == category:
            out.append(merged)
    return out


def _family_target_from_failure(spec: dict[str, Any], failed_cases: list[dict[str, Any]]):
    family_targets = spec.get("family_targets") or {}
    counts = {}
    for case in failed_cases:
        validator = case.get("validator")
        for family, target in family_targets.items():
            if validator in set(target.get("validators") or []):
                counts[family] = counts.get(family, 0) + 1
    if not counts:
        return None, None
    family = sorted(counts, key=lambda name: (-counts[name], name))[0]
    return family, dict(family_targets[family])


def _weakest_study_family(spec: dict[str, Any], study_profile: dict[str, Any]):
    scored = []
    for family, target in (spec.get("family_targets") or {}).items():
        metric = target.get("study_metric")
        if metric:
            scored.append((float(study_profile.get(metric, 0.0)), family, target))
    if not scored:
        return None, None
    _, family, target = min(scored, key=lambda x: (x[0], x[1]))
    return family, dict(target)


def _study_profile_path(seed: dict, benchmark_suite: str, study_suite: str) -> Path:
    return ROOT / "reports" / f"study-profile-brain-{seed['version']}-{benchmark_suite}-{study_suite}.json"


def strict_study_profile(seed: dict, benchmark_suite: str, force: bool = False):
    from ..checkpoint import load_entry
    from ..training.runtime import best_device
    from .study_exam import study_microbenchmark, study_suite_id
    study_suite = study_suite_id()
    path = _study_profile_path(seed, benchmark_suite, study_suite)
    if path.exists() and not force:
        value = load_json(path)
        if isinstance(value, dict) and value.get("_study_suite_id") == study_suite and value.get("_benchmark_suite_id") == benchmark_suite:
            return value, path
    model, _, tokenizer = load_entry(seed, device=best_device())
    value = study_microbenchmark(model, tokenizer)
    value["_study_suite_id"] = study_suite
    value["_benchmark_suite_id"] = benchmark_suite
    value["_seed_version"] = seed["version"]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, value)
    return value, path


@dataclass
class CapabilityNeed:
    capability: str
    metric: str
    score: float
    target: float
    gap: float
    priority: float
    recipe: str
    implementation: str
    trainable: bool
    critical_count: int
    focus_family: str | None
    focus_metric: str | None
    focus_reason: str | None
    corpus_aliases: list[str]


def capability_snapshot(force_baseline: bool = False) -> dict[str, Any]:
    seed = get_seed_entry()
    if not seed:
        raise RuntimeError("No ACTIVE/LAB seed available.")

    metrics, baseline_path = strict_baseline(seed, force=force_baseline)
    cfg = load_night_config()
    capabilities = _capability_registry()
    critical_by_category = _critical_failures_by_category(metrics)

    needs: list[CapabilityNeed] = []
    for capability, spec in (cfg.get("curriculum") or {}).items():
        metric = str(spec["metric"])
        target = float(PROMOTION_THRESHOLDS.get(metric, 1.0))
        score = float(metrics.get(metric, 0.0))
        gap = max(0.0, target - score)
        cap_meta = capabilities.get(capability, {})
        trainable = bool(cap_meta.get("trainable", True))
        implementation = str(cap_meta.get("implementation", "neural"))

        critical_category = spec.get("critical_category")
        failed_cases = _critical_cases(metrics, critical_category)
        critical_count = len(failed_cases)
        critical_pressure = critical_count * float(spec.get("critical_weight", 0.0))
        priority = gap * float(spec.get("priority_weight", 1.0)) + critical_pressure

        focus_family = None
        focus_target = None
        focus_reason = None
        if failed_cases:
            focus_family, focus_target = _family_target_from_failure(spec, failed_cases)
            if focus_target:
                focus_reason = "critical_failure"
        if not focus_target and gap > 0 and spec.get("family_targets"):
            study_profile, _ = strict_study_profile(seed, str(metrics.get("suite_id")))
            focus_family, focus_target = _weakest_study_family(spec, study_profile)
            if focus_target:
                focus_reason = "weakest_family"

        needs.append(CapabilityNeed(
            capability=capability, metric=metric, score=score, target=target, gap=gap,
            priority=priority, recipe=str(spec["recipe"]), implementation=implementation,
            trainable=trainable, critical_count=critical_count, focus_family=focus_family,
            focus_metric=(focus_target or {}).get("study_metric"), focus_reason=focus_reason,
            corpus_aliases=list((focus_target or {}).get("corpus_aliases") or []),
        ))

    needs.sort(key=lambda item: (-item.priority, -item.gap, item.capability))
    verified_count = _verified_experience_count()

    snapshot = {
        "created_at": _utcnow(),
        "seed_version": seed["version"],
        "seed_slot": "lab" if seed.get("status") == "lab" else "active",
        "suite_id": metrics.get("suite_id"),
        "baseline_path": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
        "score": metrics.get("score"),
        "promotion_eligible": metrics.get("promotion_eligible"),
        "critical_failures": metrics.get("critical_failures", []),
        "critical_failures_by_category": critical_by_category,
        "capabilities": [asdict(item) for item in needs],
        "verified_experiences_waiting": verified_count,
        "verified_experience_training_enabled": bool(
            (cfg.get("verified_experiences") or {}).get("automatic_training_enabled", False)
        ),
        "deterministic_capabilities": [
            name for name, meta in capabilities.items()
            if not bool(meta.get("trainable", True))
        ],
    }
    LATEST_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(LATEST_PLAN_PATH, snapshot)
    return snapshot


def choose_lesson(snapshot: dict[str, Any], attempted: set[str] | None = None) -> dict[str, Any] | None:
    attempted = attempted or set()
    for item in snapshot.get("capabilities", []):
        if not item.get("trainable", True):
            continue
        if float(item.get("gap", 0.0)) <= 0 and int(item.get("critical_count", 0)) <= 0:
            continue
        attempt_key = f"{item['capability']}:{item.get('focus_family') or '*'}"
        # Backward-compatible semantics:
        # - callers may mark an entire capability as attempted ("conversation")
        # - V3 sessions may mark only a specific family ("conversation:greeting")
        if item["capability"] in attempted or attempt_key in attempted:
            continue
        return item
    return None


def choose_lifelong_lesson(
    snapshot: dict[str, Any],
    attempted: set[str] | None = None,
    *,
    prefer_curriculum: bool = False,
):
    attempted = attempted or set()
    hard = []
    hard_nodes: set[str] = set()
    for raw in snapshot.get("capabilities", []):
        if not raw.get("trainable", True):
            continue
        if float(raw.get("gap", 0.0)) <= 0 and int(raw.get("critical_count", 0)) <= 0:
            continue
        lesson = enrich_legacy_lesson(raw)
        if lesson.get("plateau"):
            continue
        key = lesson_attempt_key(lesson)
        if key in attempted:
            continue
        if lesson.get("curriculum_node"):
            hard_nodes.add(str(lesson["curriculum_node"]))
        hard.append(lesson)

    # Give each unresolved hard family a turn before repeatedly attacking the same
    # one: fewer historical attempts wins, then the original hard-gate priority.
    hard.sort(
        key=lambda item: (
            int(item.get("node_attempts", 0)),
            -float(item.get("priority", 0.0)),
            str(item.get("capability")),
        )
    )
    curriculum = curriculum_lessons(attempted, exclude_nodes=hard_nodes)

    if prefer_curriculum and curriculum:
        return curriculum[0]
    if hard:
        return hard[0]
    return curriculum[0] if curriculum else None

def _stop_file(cfg: dict[str, Any]) -> Path:
    value = str(cfg.get("stop_file") or ".butterfly/STOP_NIGHT_STUDY")
    return ROOT / Path(value)


def _resource_check(cfg: dict[str, Any], started_monotonic: float, max_minutes: float) -> tuple[bool, str]:
    if _stop_file(cfg).exists():
        return False, "stop_file"

    elapsed_minutes = (time.monotonic() - started_monotonic) / 60.0
    if max_minutes > 0 and elapsed_minutes >= max_minutes:
        return False, "time_budget"

    free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
    if free_gb < float(cfg.get("min_free_disk_gb", 2.0)):
        return False, f"low_disk:{free_gb:.2f}GB"

    return True, "ok"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _run_deliberate_block():
    # Import lazily so a night session can inspect/plan without touching training.
    from ..deliberate import command_prepare, command_build, command_train, command_evaluate
    command_prepare()
    command_build()
    command_train()
    command_evaluate()


def _append_history(report: dict[str, Any]):
    history = load_json(HISTORY_PATH, {"schema_version": 1, "sessions": []})
    if not isinstance(history, dict):
        history = {"schema_version": 1, "sessions": []}
    rows = history.setdefault("sessions", [])
    rows.append(report)
    # Bounded operational history. Detailed experiment history remains elsewhere.
    history["sessions"] = rows[-100:]
    save_json(HISTORY_PATH, history)


def print_plan(snapshot: dict[str, Any]):
    print("ButterflyAI Autonomous Night Study plan")
    print(f"Seed             : {snapshot['seed_version']} ({snapshot['seed_slot']})")
    print(f"Suite            : {snapshot['suite_id']}")
    print(f"System score     : {float(snapshot.get('score') or 0):.4f}")
    print(f"Critical failures: {len(snapshot.get('critical_failures') or [])}")
    print(f"Verified backlog : {snapshot.get('verified_experiences_waiting', 0)} experience(s)")
    print("")
    print("Trainable capability backlog:")
    for item in snapshot.get("capabilities", []):
        marker = "NEEDS STUDY" if item["trainable"] and (item["gap"] > 0 or item.get("critical_count", 0) > 0) else "OK"
        print(
            f"  {item['capability']:20s} {item['score']:.4f}/{item['target']:.4f} "
            f"gap={item['gap']:.4f} critical={int(item.get('critical_count',0))} "
            f"priority={item['priority']:.4f} -> {marker}"
        )
        if item.get("focus_family"):
            print(f"      focus: {item['focus_family']} | {item.get('focus_metric')} | {item.get('focus_reason')}")
    print("")
    print("Deterministic / excluded from neural curriculum:")
    for name in snapshot.get("deterministic_capabilities", []):
        print(f"  - {name}")


def run_night_study(
    *,
    max_blocks: int | None = None,
    max_minutes: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg = load_night_config()
    if not bool(cfg.get("enabled", True)):
        raise RuntimeError("Night Study is disabled in config/night_study.json")

    current = load_current_experiment()
    terminal = {"promoted", "lab_accepted", "rejected", "cancelled"}
    if current and current.get("status") not in terminal:
        raise RuntimeError(
            f"Cannot start Night Study: experiment {current.get('experiment_id')} "
            f"is still {current.get('status')}."
        )

    max_blocks = int(cfg.get("max_blocks_per_session", 0) if max_blocks is None else max_blocks)
    max_minutes = float(cfg.get("max_minutes_per_session", 0) if max_minutes is None else max_minutes)
    if max_blocks < 0:
        raise ValueError("max_blocks must be >= 0 (0 means unlimited)")
    if max_minutes < 0:
        raise ValueError("max_minutes must be >= 0 (0 means unlimited)")

    first_snapshot = capability_snapshot(force_baseline=False)
    first_lifelong = strict_dynamic_diagnostic(force=False)
    if dry_run:
        print_plan(first_snapshot)
        print("\nLifelong curriculum diagnostic:")
        print(f"  Dynamic suite : {first_lifelong.get('dynamic_suite_id')}")
        print(f"  Nodes assessed: {len(first_lifelong.get('results') or {})}")
        ready = ((first_lifelong.get("curriculum") or {}).get("next_trainable_with_verified_material") or [])[:8]
        if ready:
            print("  Internal curriculum weaknesses:")
            for item in ready:
                mastery = "unknown" if item.get("mastery") is None else f"{float(item['mastery']):.3f}"
                print(f"    - {item['id']}: mastery={mastery} priority={float(item['priority']):.3f}")
        lesson = choose_lifelong_lesson(first_snapshot)
        if lesson:
            print("")
            print(
                f"Next autonomous lesson: {lesson['capability']} "
                f"using recipe {lesson['recipe']}"
            )
        else:
            print("\nNo safe internal lesson currently needs study.")
        return {
            "dry_run": True,
            "snapshot": first_snapshot,
            "lifelong": first_lifelong,
            "next_lesson": lesson,
        }

    stop_path = _stop_file(cfg)
    # A stale stop request should not make every future session silently exit.
    stop_path.unlink(missing_ok=True)

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = ROOT / "logs" / f"night-study-{session_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = _utcnow()
    started_mono = time.monotonic()
    attempted: set[str] = set()
    blocks: list[dict[str, Any]] = []
    stop_reason = "completed"

    with log_path.open("a", encoding="utf-8") as log:
        tee_out = _Tee(sys.stdout, log)
        tee_err = _Tee(sys.stderr, log)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("=" * 72)
            print(" ButterflyAI LIFELONG NIGHT STUDY V4")
            print("=" * 72)
            print(f"Session          : {session_id}")
            print(f"Maximum blocks   : {'unlimited' if max_blocks == 0 else max_blocks}")
            print(f"Maximum minutes  : {'unlimited' if max_minutes == 0 else f'{max_minutes:g}'}")
            print(f"Stop file        : {stop_path}")
            print("Deterministic skills are excluded from neural curriculum.")
            print("Verified experiences remain provenance-gated; raw web/data is never auto-trained.")
            print("")

            block_index = 0
            while max_blocks == 0 or block_index < max_blocks:
                block_index += 1
                allowed, reason = _resource_check(cfg, started_mono, max_minutes)
                if not allowed:
                    stop_reason = reason
                    print(f"Night Study stopping before block {block_index}: {reason}")
                    break

                snapshot = capability_snapshot(force_baseline=False)
                print_plan(snapshot)
                lesson = choose_lifelong_lesson(snapshot, attempted, prefer_curriculum=(block_index % 2 == 0))
                if lesson is None:
                    stop_reason = "research_required" if research_needed() else "curriculum_exhausted"
                    print(
                        "No safe internal lesson remains. "
                        + ("External verified research is the next dependency." if stop_reason == "research_required" else "Current curriculum is exhausted.")
                    )
                    break

                attempted.add(lesson_attempt_key(lesson))

                current = load_current_experiment()
                if current:
                    clear_terminal_experiment()

                focus_target = focus_target_for_lesson(lesson)
                experiment = create_experiment(lesson["recipe"], focus_target=focus_target)
                print("")
                print("-" * 72)
                print(
                    f"BLOCK {block_index}/{('∞' if max_blocks == 0 else max_blocks)}: {lesson['capability']}/{lesson.get('focus_family') or 'general'} "
                    f"-> {lesson['recipe']}"
                )
                print(
                    f"Target {experiment['target_version']} from "
                    f"{experiment['seed_version']} ({experiment['seed_slot']})"
                )
                print("-" * 72)

                block_record = {
                    "block": block_index,
                    "capability": lesson["capability"],
                    "focus_family": lesson.get("focus_family"),
                    "focus_metric": lesson.get("focus_metric"),
                    "focus_reason": lesson.get("focus_reason"),
                    "curriculum_node": lesson.get("curriculum_node"),
                    "dynamic_family": lesson.get("dynamic_family"),
                    "strategy_id": lesson.get("strategy_id"),
                    "recipe": lesson["recipe"],
                    "experiment_id": experiment["experiment_id"],
                    "target_version": experiment["target_version"],
                    "seed_version": experiment["seed_version"],
                    "started_at": _utcnow(),
                    "status": "running",
                }
                try:
                    _run_deliberate_block()
                    final = load_current_experiment() or {}
                    block_record.update({
                        "finished_at": _utcnow(),
                        "status": final.get("status", "unknown"),
                        "final_score": final.get("final_score"),
                        "evaluation_report": final.get("evaluation_report"),
                    })
                    report_rel = final.get("evaluation_report")
                    if report_rel:
                        comparison = load_json(ROOT / report_rel, {})
                        if isinstance(comparison, dict):
                            block_record["lifelong_acceptance"] = comparison.get("lifelong_acceptance")
                    try:
                        record_block_outcome(lesson, block_record)
                    except Exception as memory_exc:
                        block_record["strategy_memory_warning"] = f"{type(memory_exc).__name__}: {memory_exc}"
                        print(f"Strategy-memory warning (training result remains valid): {memory_exc}")
                    print(
                        f"BLOCK {block_index} result: {block_record['status']} "
                        f"score={block_record.get('final_score')}"
                    )
                except BaseException as exc:
                    block_record.update({
                        "finished_at": _utcnow(),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    blocks.append(block_record)
                    stop_reason = "error"
                    print(f"Night Study block failed safely: {block_record['error']}")
                    break

                blocks.append(block_record)

                if (
                    block_record["status"] == "promoted"
                    and bool(cfg.get("stop_after_active_promotion", True))
                ):
                    stop_reason = "active_promoted"
                    print("ACTIVE promotion achieved; configured stop-after-promotion requested a safe stop.")
                    break

            final_snapshot = capability_snapshot(force_baseline=False)
            final_lifelong = strict_dynamic_diagnostic(force=False)
            print("")
            print("=" * 72)
            print(" NIGHT STUDY SESSION COMPLETE")
            print("=" * 72)
            print(f"Stop reason: {stop_reason}")
            print(f"Blocks run : {len(blocks)}")
            print(f"Final seed : {final_snapshot['seed_version']} ({final_snapshot['seed_slot']})")
            print(f"Final score: {float(final_snapshot.get('score') or 0):.4f}")

    report = {
        "schema_version": 2,
        "session_id": session_id,
        "started_at": started_wall,
        "finished_at": _utcnow(),
        "stop_reason": stop_reason,
        "max_blocks": max_blocks,
        "max_minutes": max_minutes,
        "blocks": blocks,
        "initial_snapshot": first_snapshot,
        "initial_lifelong": first_lifelong,
        "final_snapshot": final_snapshot,
        "final_lifelong": final_lifelong,
        "log_path": str(log_path.relative_to(ROOT)).replace("\\", "/"),
    }
    LATEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(LATEST_REPORT_PATH, report)
    _append_history(report)
    return report
