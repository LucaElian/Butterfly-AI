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
        critical_count = int(critical_by_category.get(critical_category, 0)) if critical_category else 0
        critical_pressure = critical_count * float(spec.get("critical_weight", 0.0))
        priority = gap * float(spec.get("priority_weight", 1.0)) + critical_pressure
        needs.append(CapabilityNeed(
            capability=capability,
            metric=metric,
            score=score,
            target=target,
            gap=gap,
            priority=priority,
            recipe=str(spec["recipe"]),
            implementation=implementation,
            trainable=trainable,
            critical_count=critical_count,
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
        if item["capability"] in attempted:
            continue
        return item
    return None


def _stop_file(cfg: dict[str, Any]) -> Path:
    value = str(cfg.get("stop_file") or ".butterfly/STOP_NIGHT_STUDY")
    return ROOT / Path(value)


def _resource_check(cfg: dict[str, Any], started_monotonic: float, max_minutes: float) -> tuple[bool, str]:
    if _stop_file(cfg).exists():
        return False, "stop_file"

    elapsed_minutes = (time.monotonic() - started_monotonic) / 60.0
    if elapsed_minutes >= max_minutes:
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

    max_blocks = int(max_blocks or cfg.get("max_blocks_per_session", 2))
    max_minutes = float(max_minutes or cfg.get("max_minutes_per_session", 180))
    if max_blocks < 1:
        raise ValueError("max_blocks must be >= 1")
    if max_minutes <= 0:
        raise ValueError("max_minutes must be > 0")

    first_snapshot = capability_snapshot(force_baseline=False)
    if dry_run:
        print_plan(first_snapshot)
        lesson = choose_lesson(first_snapshot)
        if lesson:
            print("")
            print(
                f"Next autonomous lesson: {lesson['capability']} "
                f"using recipe {lesson['recipe']}"
            )
        else:
            print("\nNo trainable hard-gate deficit currently needs a lesson.")
        return {
            "dry_run": True,
            "snapshot": first_snapshot,
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
            print(" ButterflyAI AUTONOMOUS NIGHT STUDY V1")
            print("=" * 72)
            print(f"Session          : {session_id}")
            print(f"Maximum blocks   : {max_blocks}")
            print(f"Maximum minutes  : {max_minutes:g}")
            print(f"Stop file        : {stop_path}")
            print("Deterministic skills are excluded from neural curriculum.")
            print("Verified experiences are tracked but NOT auto-trained in V1.")
            print("")

            for block_index in range(1, max_blocks + 1):
                allowed, reason = _resource_check(cfg, started_mono, max_minutes)
                if not allowed:
                    stop_reason = reason
                    print(f"Night Study stopping before block {block_index}: {reason}")
                    break

                snapshot = capability_snapshot(force_baseline=False)
                print_plan(snapshot)
                lesson = choose_lesson(snapshot, attempted)
                if lesson is None:
                    stop_reason = "no_trainable_deficit"
                    print("No unattempted trainable hard-gate deficit remains for this session.")
                    break

                if not bool(cfg.get("retry_same_capability_in_session", False)):
                    attempted.add(lesson["capability"])

                current = load_current_experiment()
                if current:
                    clear_terminal_experiment()

                experiment = create_experiment(lesson["recipe"])
                print("")
                print("-" * 72)
                print(
                    f"BLOCK {block_index}/{max_blocks}: {lesson['capability']} "
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
                    print("ACTIVE promotion achieved; stopping this V1 session safely.")
                    break

            final_snapshot = capability_snapshot(force_baseline=False)
            print("")
            print("=" * 72)
            print(" NIGHT STUDY SESSION COMPLETE")
            print("=" * 72)
            print(f"Stop reason: {stop_reason}")
            print(f"Blocks run : {len(blocks)}")
            print(f"Final seed : {final_snapshot['seed_version']} ({final_snapshot['seed_slot']})")
            print(f"Final score: {float(final_snapshot.get('score') or 0):.4f}")

    report = {
        "schema_version": 1,
        "session_id": session_id,
        "started_at": started_wall,
        "finished_at": _utcnow(),
        "stop_reason": stop_reason,
        "max_blocks": max_blocks,
        "max_minutes": max_minutes,
        "blocks": blocks,
        "initial_snapshot": first_snapshot,
        "final_snapshot": capability_snapshot(force_baseline=False),
        "log_path": str(log_path.relative_to(ROOT)).replace("\\", "/"),
    }
    LATEST_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(LATEST_REPORT_PATH, report)
    _append_history(report)
    return report
