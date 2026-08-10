from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "pipeline.json"
STATE_PATH = ROOT / ".butterfly" / "pipeline_state.json"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"

STAGES = (
    ("prepare", "01 PREPARE"),
    ("build_dataset", "02 BUILD DATASET"),
    ("train", "03 TRAIN"),
    ("evaluate_and_promote", "04 EVALUATE AND PROMOTE"),
)

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text: str):
        print(text, end="", flush=True)
        for f in self.files:
            f.write(text)
            f.flush()

def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def load_recipe():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing permanent pipeline config: {CONFIG_PATH}\n"
            "Reinstall the permanent pipeline infrastructure."
        )
    recipe = _load_json(CONFIG_PATH, {})
    if recipe.get("schema_version") != 1:
        raise RuntimeError("Unsupported pipeline schema.")
    return recipe

def _fresh_state(recipe):
    return {
        "schema_version": 1,
        "target_brain": recipe.get("target_brain"),
        "benchmark_suite": recipe.get("benchmark_suite"),
        "updated_at": time.time(),
        "stages": {name: "pending" for name, _ in STAGES},
        "last_error": None,
        "last_log": None,
    }

def load_state(recipe):
    state = _load_json(STATE_PATH, None)
    if not state or state.get("target_brain") != recipe.get("target_brain"):
        state = _fresh_state(recipe)
        save_state(state)
    return state

def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.time()
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)

def _configured_command(recipe, stage):
    raw = (recipe.get("stages") or {}).get(stage)
    if not raw:
        return None
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RuntimeError(f"Stage {stage!r} must be null or a JSON list of arguments.")
    return raw

def _require_target(recipe):
    target = recipe.get("target_brain")
    if not target:
        raise RuntimeError(
            "No target brain is configured yet.\n"
            "The permanent pipeline infrastructure is installed correctly, but it is intentionally idle.\n"
            "The v0.00052 update will configure the SAME 01-04 files; no new version-numbered BATs are needed."
        )
    return str(target)

def _open_logs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = LOGS_DIR / f"pipeline-{stamp}.log"
    latest = LOGS_DIR / "latest.log"
    # latest is overwritten for each run and remains useful even after a crash.
    return run_path, latest, run_path.open("w", encoding="utf-8"), latest.open("w", encoding="utf-8")

def _write_summary(recipe, state, run_log):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = recipe.get("target_brain") or "unconfigured"
    lines = [
        "ButterflyAI permanent pipeline summary",
        "======================================",
        f"Target brain : {target}",
        f"Benchmark    : v{recipe.get('benchmark_suite')}",
        f"Preset       : {recipe.get('preset')}",
        f"Run log      : {run_log}",
        "",
        "Stages:",
    ]
    for name, label in STAGES:
        lines.append(f"- {label}: {state['stages'].get(name, 'pending').upper()}")
    if state.get("last_error"):
        lines += ["", "Last error:", str(state["last_error"])]
    text = "\n".join(lines) + "\n"
    (REPORTS_DIR / "latest-summary.txt").write_text(text, encoding="utf-8")
    if target != "unconfigured":
        (REPORTS_DIR / f"v{target}-summary.txt").write_text(text, encoding="utf-8")

def _run_stage(recipe, state, stage, tee):
    target = _require_target(recipe)
    command = _configured_command(recipe, stage)
    if command is None:
        raise RuntimeError(
            f"Target v{target} is configured but stage {stage!r} has no command. "
            "The target recipe is incomplete."
        )

    label = dict(STAGES)[stage]
    tee.write("\n" + "=" * 72 + "\n")
    tee.write(f"{label} | target v{target}\n")
    tee.write("=" * 72 + "\n")
    state["stages"][stage] = "running"
    state["last_error"] = None
    save_state(state)

    env = os.environ.copy()
    env["BUTTERFLY_PIPELINE_TARGET"] = target
    env["BUTTERFLY_PIPELINE_STAGE"] = stage
    env["BUTTERFLY_PIPELINE_PRESET"] = str(recipe.get("preset", "auto"))

    proc = subprocess.Popen(
        [sys.executable, *command],
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
        state["stages"][stage] = "failed"
        state["last_error"] = f"{label} exited with code {code}"
        save_state(state)
        raise RuntimeError(state["last_error"])

    state["stages"][stage] = "complete"
    save_state(state)
    tee.write(f"\n[{label}] OK\n")

def _first_incomplete(state):
    for name, _ in STAGES:
        if state["stages"].get(name) != "complete":
            return name
    return None

def _reset_from(state, stage):
    names = [n for n, _ in STAGES]
    idx = names.index(stage)
    for n in names[idx:]:
        state["stages"][n] = "pending"
    state["last_error"] = None
    save_state(state)

def run(mode: str, chosen_stage: str | None = None):
    recipe = load_recipe()
    target = _require_target(recipe)
    state = load_state(recipe)

    if mode == "auto":
        # A normal automatic run is a new run from PREPARE.
        _reset_from(state, STAGES[0][0])
        start_stage = STAGES[0][0]
    elif mode == "paused":
        _reset_from(state, STAGES[0][0])
        start_stage = STAGES[0][0]
    elif mode == "resume":
        start_stage = _first_incomplete(state)
        if start_stage is None:
            print(f"Target v{target}: all four stages are already complete.")
            return 0
    elif mode == "stage":
        if chosen_stage not in dict(STAGES):
            raise RuntimeError(f"Unknown stage: {chosen_stage}")
        # Running one stage explicitly invalidates later stage statuses.
        _reset_from(state, chosen_stage)
        start_stage = chosen_stage
    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    run_path, latest_path, run_file, latest_file = _open_logs()
    tee = Tee(run_file, latest_file)
    state["last_log"] = str(run_path.relative_to(ROOT))
    save_state(state)
    try:
        tee.write("ButterflyAI permanent pipeline\n")
        tee.write(f"Target: v{target} | benchmark: v{recipe.get('benchmark_suite')} | mode: {mode}\n")
        tee.write(f"Log: {run_path}\n")

        names = [n for n, _ in STAGES]
        if mode == "stage":
            to_run = [start_stage]
        else:
            to_run = names[names.index(start_stage):]

        for index, stage in enumerate(to_run):
            _run_stage(recipe, state, stage, tee)
            if mode == "paused" and index < len(to_run) - 1:
                answer = input("\nStage complete. Continue? [Y/N]: ").strip().lower()
                if answer not in ("y", "yes", "s", "si", "sí"):
                    tee.write("\nPipeline paused by user. Use REANUDAR later.\n")
                    break

        tee.write("\nPipeline finished without an unhandled stage error.\n")
        return 0
    except Exception as exc:
        state["last_error"] = str(exc)
        save_state(state)
        tee.write(f"\nPIPELINE STOPPED: {exc}\n")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(run_path, LOGS_DIR / "latest-error.log")
        except Exception:
            pass
        return 1
    finally:
        _write_summary(recipe, state, run_path)
        run_file.close()
        latest_file.close()

def status():
    recipe = load_recipe()
    state = load_state(recipe)
    print(json.dumps({
        "target_brain": recipe.get("target_brain"),
        "benchmark_suite": recipe.get("benchmark_suite"),
        "preset": recipe.get("preset"),
        "allow_sleep_learning": recipe.get("allow_sleep_learning", False),
        "pipeline_state": state,
    }, indent=2, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(prog="python -m butterfly.pipeline")
    parser.add_argument("--mode", choices=("auto", "paused", "resume", "stage", "status"), required=True)
    parser.add_argument("--stage", choices=tuple(n for n, _ in STAGES))
    args = parser.parse_args()
    if args.mode == "status":
        status()
        return
    if args.mode == "stage" and not args.stage:
        parser.error("--stage is required with --mode stage")
    raise SystemExit(run(args.mode, args.stage))

if __name__ == "__main__":
    main()
