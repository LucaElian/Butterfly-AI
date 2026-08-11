from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import ROOT, PIPELINE_CONFIG_PATH, load_json
from .experiments import ensure_experiment, load_current_experiment, load_recipe

STATE_PATH = ROOT / ".butterfly" / "pipeline_state.json"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"

STAGES = (
    ("prepare", "01 PREPARE"),
    ("build_dataset", "02 BUILD DATASET"),
    ("train", "03 TRAIN"),
    ("evaluate_and_promote", "04 EVALUATE AND PROMOTE"),
)
STAGE_NAMES = tuple(name for name, _ in STAGES)
STATE_SCHEMA = 2


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text: str):
        print(text, end="", flush=True)
        for handle in self.files:
            handle.write(text)
            handle.flush()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _sha256_bytes(raw.encode("utf-8"))


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config():
    cfg = load_json(PIPELINE_CONFIG_PATH, {})
    if cfg.get("schema_version") != 2:
        raise RuntimeError("Unsupported pipeline schema. Run SETUP_WINDOWS.bat or the refactor installer.")
    return cfg


def _fresh_state(exp):
    return {
        "schema_version": STATE_SCHEMA,
        "experiment_id": exp["experiment_id"],
        "updated_at": time.time(),
        "stages": {
            name: {"status": "pending", "signature": None, "completed_at": None, "log": None}
            for name in STAGE_NAMES
        },
        "last_error": None,
    }


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.time()
    tmp = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def load_state(exp):
    state = load_json(STATE_PATH)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != STATE_SCHEMA
        or state.get("experiment_id") != exp.get("experiment_id")
    ):
        state = _fresh_state(exp)
        save_state(state)
    return state


def _configured_command(cfg, stage):
    raw = (cfg.get("stages") or {}).get(stage)
    if not isinstance(raw, list) or not raw or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"Pipeline stage {stage!r} has no valid command.")
    return raw


def _source_tree_fingerprint(paths: list[Path]) -> str:
    rows = []
    for base in paths:
        if base.is_file():
            rows.append((str(base.relative_to(ROOT)), _sha256_file(base)))
        elif base.is_dir():
            for path in sorted(base.rglob("*.py")):
                rows.append((str(path.relative_to(ROOT)), _sha256_file(path)))
    return _sha256_json(rows)


def _manifest_digest() -> str | None:
    return _sha256_file(ROOT / "data" / "corpus" / "deliberate" / "manifest.json")


def _candidate_digest(exp) -> str | None:
    from .registry import get_candidate_entry
    candidate = get_candidate_entry()
    if not candidate or candidate.get("version") != exp.get("target_version"):
        return None
    return _sha256_file(ROOT / "models" / candidate["path"])


def _stage_signature(stage: str, cfg: dict, exp: dict, recipe: dict) -> str:
    from .learning.evaluator import BENCHMARK_SUITE_ID
    from .registry import get_entry

    seed = get_entry(exp.get("seed_version"))
    seed_path = ROOT / "models" / seed["path"] if seed else None
    base = {
        "stage": stage,
        "experiment_id": exp.get("experiment_id"),
        "target_version": exp.get("target_version"),
        "seed_version": exp.get("seed_version"),
        "recipe_hash": exp.get("recipe_hash"),
        "suite_id": BENCHMARK_SUITE_ID,
        "seed_model": _sha256_file(seed_path) if seed_path else None,
    }
    if stage == "prepare":
        base["preflight_code"] = _source_tree_fingerprint([
            ROOT / "butterfly" / "deliberate.py",
            ROOT / "butterfly" / "upgrade.py",
            ROOT / "butterfly" / "experiments.py",
        ])
    elif stage == "build_dataset":
        base["corpus_code"] = _source_tree_fingerprint([
            ROOT / "butterfly" / "corpus" / "deliberate.py",
            ROOT / "butterfly" / "corpus" / "skills",
        ])
    elif stage == "train":
        base["manifest"] = _manifest_digest()
        base["trainer_code"] = _source_tree_fingerprint([
            ROOT / "butterfly" / "deliberate_trainer.py",
            ROOT / "butterfly" / "alignment_data.py",
            ROOT / "butterfly" / "training",
        ])
    elif stage == "evaluate_and_promote":
        base["candidate"] = _candidate_digest(exp)
        base["policy"] = _sha256_file(ROOT / "config" / "promotion_policy.json")
    return _sha256_json(base)


def _reset_from(state, stage):
    index = STAGE_NAMES.index(stage)
    for name in STAGE_NAMES[index:]:
        state["stages"][name] = {
            "status": "pending",
            "signature": None,
            "completed_at": None,
            "log": None,
        }
    state["last_error"] = None
    save_state(state)


def _refresh_validity(state, cfg, exp, recipe):
    terminal = exp.get("status") in {"promoted", "lab_accepted", "rejected", "cancelled"}
    for name in STAGE_NAMES:
        row = state["stages"].get(name) or {}
        if row.get("status") != "complete":
            continue
        if terminal and name == "evaluate_and_promote":
            continue
        expected = _stage_signature(name, cfg, exp, recipe)
        if row.get("signature") != expected:
            _reset_from(state, name)
            break


def _first_incomplete(state):
    for name in STAGE_NAMES:
        if (state["stages"].get(name) or {}).get("status") != "complete":
            return name
    return None


def _missing_prerequisites(state, stage: str):
    index = STAGE_NAMES.index(stage)
    return tuple(
        name
        for name in STAGE_NAMES[:index]
        if (state["stages"].get(name) or {}).get("status") != "complete"
    )


def _safe_fragment(value: str) -> str:
    value = str(value).strip().lower().replace("_", "-")
    return "".join(ch for ch in value if ch.isalnum() or ch in "-.") or "unknown"


def _open_stage_logs(stage: str, exp: dict):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target = _safe_fragment(exp.get("target_version"))
    recipe = _safe_fragment(exp.get("recipe_name"))
    stage_name = _safe_fragment(stage)
    run_path = LOGS_DIR / f"{stamp}__{stage_name}__target-v{target}__recipe-{recipe}.log"
    latest_path = LOGS_DIR / "latest.log"
    return (
        run_path,
        latest_path,
        run_path.open("w", encoding="utf-8"),
        latest_path.open("a", encoding="utf-8"),
    )


def _write_summary(exp, state):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "ButterflyAI pipeline summary",
        "============================",
        f"Experiment : {exp.get('experiment_id')}",
        f"Target     : {exp.get('target_version')}",
        f"Seed       : {exp.get('seed_version')} ({exp.get('seed_slot')})",
        f"Recipe     : {exp.get('recipe_name')}",
        f"Suite      : {exp.get('suite_id')}",
        f"Result     : {exp.get('status')}",
        "",
        "Stages:",
    ]
    for name, label in STAGES:
        row = state["stages"].get(name) or {}
        lines.append(f"- {label}: {str(row.get('status', 'pending')).upper()}")
        if row.get("log"):
            lines.append(f"  log: {row['log']}")
    if state.get("last_error"):
        lines += ["", "Last error:", str(state["last_error"])]
    (REPORTS_DIR / "latest-summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_stage(cfg, exp, recipe, state, stage):
    command = _configured_command(cfg, stage)
    label = dict(STAGES)[stage]
    run_path, latest_path, run_file, latest_file = _open_stage_logs(stage, exp)
    tee = Tee(run_file, latest_file)
    row = state["stages"][stage]
    row["status"] = "running"
    row["log"] = str(run_path.relative_to(ROOT))
    state["last_error"] = None
    save_state(state)

    env = os.environ.copy()
    env["BUTTERFLY_EXPERIMENT_ID"] = str(exp["experiment_id"])
    env["BUTTERFLY_PIPELINE_TARGET"] = str(exp["target_version"])
    env["BUTTERFLY_PIPELINE_STAGE"] = stage
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        tee.write("ButterflyAI pipeline stage\n")
        tee.write(f"Stage      : {label}\n")
        tee.write(f"Experiment : {exp['experiment_id']}\n")
        tee.write(f"Target     : {exp['target_version']}\n")
        tee.write(f"Seed       : {exp['seed_version']} ({exp['seed_slot']})\n")
        tee.write(f"Recipe     : {exp['recipe_name']}\n")
        tee.write(f"Suite      : {exp['suite_id']}\n")
        tee.write(f"Log        : {run_path}\n")
        tee.write("=" * 72 + "\n")

        proc = subprocess.Popen(
            [sys.executable, "-u", *command],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            tee.write(line)
        code = proc.wait()
        if code != 0:
            row["status"] = "failed"
            state["last_error"] = f"{label} exited with code {code}"
            save_state(state)
            raise RuntimeError(state["last_error"])

        exp = load_current_experiment() or exp
        recipe = load_recipe(exp["recipe_name"])
        row["status"] = "complete"
        row["signature"] = _stage_signature(stage, cfg, exp, recipe)
        row["completed_at"] = time.time()
        save_state(state)
        tee.write(f"\n[{label}] OK\n")
        return exp
    except Exception as exc:
        row["status"] = "failed"
        state["last_error"] = str(exc)
        save_state(state)
        tee.write(f"\nPIPELINE STOPPED: {exc}\n")
        try:
            shutil.copyfile(latest_path, LOGS_DIR / "latest-error.log")
        except Exception:
            pass
        raise
    finally:
        run_file.close()
        latest_file.close()


def run(mode: str, chosen_stage: str | None = None):
    cfg = load_config()
    exp = ensure_experiment(create=True)
    recipe = load_recipe(exp["recipe_name"])
    state = load_state(exp)
    _refresh_validity(state, cfg, exp, recipe)

    if mode in {"auto", "paused"}:
        start = _first_incomplete(state)
        if start is None:
            print("Current experiment already completed all pipeline stages.")
            print("Create a new experiment only when you intentionally choose the next recipe/objective.")
            _write_summary(exp, state)
            return 0
        to_run = STAGE_NAMES[STAGE_NAMES.index(start):]
    elif mode == "stage":
        if chosen_stage not in STAGE_NAMES:
            raise RuntimeError(f"Unknown stage: {chosen_stage}")
        missing = _missing_prerequisites(state, chosen_stage)
        if missing:
            labels = ", ".join(dict(STAGES)[name] for name in missing)
            raise RuntimeError(
                f"Cannot run {dict(STAGES)[chosen_stage]} manually; "
                f"complete prerequisite stages first: {labels}"
            )
        _reset_from(state, chosen_stage)
        to_run = (chosen_stage,)
    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "latest.log").write_text("", encoding="utf-8")

    try:
        for index, stage in enumerate(to_run):
            exp = _run_stage(cfg, exp, recipe, state, stage)
            recipe = load_recipe(exp["recipe_name"])
            if mode == "paused" and index < len(to_run) - 1:
                input("\nEtapa completada. Presiona ENTER para continuar...")
        print("\nPipeline finished without an unhandled stage error.")
        return 0
    except Exception:
        return 1
    finally:
        exp = load_current_experiment() or exp
        _write_summary(exp, state)


def pipeline_status():
    exp = load_current_experiment()
    if not exp:
        return {"experiment": None, "stages": {name: "pending" for name in STAGE_NAMES}}
    cfg = load_config()
    recipe = load_recipe(exp["recipe_name"])
    state = load_state(exp)
    _refresh_validity(state, cfg, exp, recipe)
    return {
        "experiment_id": exp.get("experiment_id"),
        "target": exp.get("target_version"),
        "recipe": exp.get("recipe_name"),
        "status": exp.get("status"),
        "stages": {name: (state["stages"].get(name) or {}).get("status", "pending") for name in STAGE_NAMES},
        "last_error": state.get("last_error"),
    }


def stage_menu():
    print()
    for index, (name, label) in enumerate(STAGES, 1):
        print(f"[{index}] {label}")
    raw = input("Etapa: ").strip()
    try:
        chosen = STAGE_NAMES[int(raw) - 1]
    except (ValueError, IndexError):
        print("Opcion invalida.")
        return 2
    return run("stage", chosen)


def main():
    parser = argparse.ArgumentParser(prog="python -m butterfly.pipeline")
    parser.add_argument(
        "--mode",
        choices=("auto", "paused", "stage", "stage-menu", "status"),
        required=True,
    )
    parser.add_argument("--stage", choices=STAGE_NAMES)
    args = parser.parse_args()

    if args.mode == "status":
        print(json.dumps(pipeline_status(), indent=2, ensure_ascii=False))
        return
    if args.mode == "stage-menu":
        raise SystemExit(stage_menu())
    if args.mode == "stage" and not args.stage:
        parser.error("--stage is required with --mode stage")
    raise SystemExit(run(args.mode, args.stage))


if __name__ == "__main__":
    main()
