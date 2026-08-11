from __future__ import annotations

import argparse
import json

from .config import ROOT, project_relpath
from .experiments import ensure_experiment, load_recipe, mark_experiment_status


def _context():
    exp = ensure_experiment(create=True)
    recipe = load_recipe(exp["recipe_name"])
    if exp.get("recipe_hash") is None:
        raise RuntimeError("Experiment has no recipe hash.")
    return exp, recipe


def command_prepare():
    from .learning.evaluator import BENCHMARK_SUITE_ID
    from .upgrade import prepare_target

    exp, recipe = _context()
    if exp.get("suite_id") != BENCHMARK_SUITE_ID:
        raise RuntimeError("Evaluator changed after experiment creation. Create a fresh experiment.")
    print("ButterflyAI deliberate preflight")
    print("No training occurs in this stage.")
    prepare_target(exp["target_version"], seed_version=exp["seed_version"])
    mark_experiment_status("prepared")


def command_build():
    from .corpus.deliberate import build_corpus

    exp, recipe = _context()
    print("ButterflyAI deliberate corpus build")
    build_corpus(exp, recipe)
    mark_experiment_status("dataset_ready")


def command_train():
    from .deliberate_trainer import train_candidate

    exp, recipe = _context()
    print("ButterflyAI deliberate training")
    print("USER tokens are context; loss is computed only on Butterfly answer tokens.")
    train_candidate(exp, recipe)
    mark_experiment_status("candidate_ready")


def command_evaluate():
    from .upgrade import evaluate_candidate

    exp, recipe = _context()
    result, report_path, metrics = evaluate_candidate(exp["target_version"], recipe)
    status = {
        "PROMOTED": "promoted",
        "LAB_ACCEPTED": "lab_accepted",
        "REJECTED": "rejected",
    }[result]
    mark_experiment_status(
        status,
        evaluation_report=project_relpath(report_path),
        final_score=metrics.get("score"),
    )

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    compact = {
        "experiment_id": exp["experiment_id"],
        "target_version": exp["target_version"],
        "seed_version": exp["seed_version"],
        "recipe_name": exp["recipe_name"],
        "result": result,
        "score": metrics.get("score"),
        "suite_id": metrics.get("suite_id"),
        "critical_pass_rate": metrics.get("critical_pass_rate"),
        "promotion_eligible": metrics.get("promotion_eligible"),
        "promotion_blockers": metrics.get("promotion_blockers", []),
    }
    text = json.dumps(compact, indent=2, ensure_ascii=False)
    (reports / "latest-evaluation.json").write_text(text, encoding="utf-8")
    (reports / f"brain-{exp['target_version']}-evaluation.json").write_text(text, encoding="utf-8")
    print(f"Final result: {result}")


def main():
    parser = argparse.ArgumentParser(prog="python -m butterfly.deliberate")
    parser.add_argument("command", choices=("prepare", "build", "train", "evaluate"))
    args = parser.parse_args()
    {
        "prepare": command_prepare,
        "build": command_build,
        "train": command_train,
        "evaluate": command_evaluate,
    }[args.command]()


if __name__ == "__main__":
    main()
