from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ROOT

RECIPE_PATH = ROOT / "config" / "deliberate_recipe.json"
PIPELINE_PATH = ROOT / "config" / "pipeline.json"


def load_recipe() -> dict:
    if not RECIPE_PATH.exists():
        raise FileNotFoundError(RECIPE_PATH)
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != 1:
        raise RuntimeError("Unsupported deliberate recipe schema.")
    return recipe


def _assert_pipeline_matches(recipe: dict):
    cfg = json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
    if str(cfg.get("target_brain")) != str(recipe.get("target_brain")):
        raise RuntimeError("pipeline.json target does not match deliberate_recipe.json target.")
    if str(cfg.get("benchmark_suite")) != str(recipe.get("benchmark_suite")):
        raise RuntimeError("pipeline.json benchmark suite does not match deliberate recipe.")


def command_prepare(recipe: dict):
    from .learning.evaluator import BENCHMARK_SUITE_VERSION
    from .upgrade import prepare_target

    _assert_pipeline_matches(recipe)
    if str(recipe["benchmark_suite"]) != str(BENCHMARK_SUITE_VERSION):
        raise RuntimeError(
            f"Recipe requests benchmark v{recipe['benchmark_suite']}, but evaluator is v{BENCHMARK_SUITE_VERSION}."
        )
    print(f"ButterflyAI v{recipe['target_brain']} deliberate preflight")
    print("No training occurs in this stage.")
    prepare_target(str(recipe["target_brain"]), expected_active=str(recipe["expected_active"]))


def command_build(recipe: dict):
    from .corpus.deliberate import build_corpus

    _assert_pipeline_matches(recipe)
    print(f"ButterflyAI v{recipe['target_brain']} deliberate corpus")
    print("Offline generation only: NO Internet and NO Qwen/teacher model.")
    build_corpus(str(recipe["target_brain"]))


def command_train(recipe: dict):
    from .deliberate_trainer import train_candidate

    _assert_pipeline_matches(recipe)
    print(f"ButterflyAI v{recipe['target_brain']} deliberate training")
    print("Seed = accepted active brain + same accepted tokenizer.")
    print("USER = context; loss ONLY on Butterfly answer tokens.")
    train_candidate(str(recipe["target_brain"]), recipe)


def command_evaluate(recipe: dict):
    from .registry import get_active_entry
    from .upgrade import compare_and_promote

    _assert_pipeline_matches(recipe)
    before = get_active_entry()
    if not before:
        raise RuntimeError("No active baseline before evaluation.")
    baseline_version = before["version"]
    target = str(recipe["target_brain"])
    print(f"ButterflyAI v{target} strict comparison")
    print("Score improvement alone is NOT enough: every hard gate must pass.")
    result = compare_and_promote(target)

    comparison = ROOT / "benchmarks" / (
        f"comparison-v{baseline_version}-vs-v{target}-suite-v{recipe['benchmark_suite']}.json"
    )
    if comparison.exists():
        payload = json.loads(comparison.read_text(encoding="utf-8"))
        baseline = payload.get("baseline", {})
        candidate = payload.get("candidate", {})
        compact = {
            "target_version": target,
            "baseline_version": baseline_version,
            "benchmark_suite": recipe["benchmark_suite"],
            "result": "PROMOTED" if result else "REJECTED",
            "baseline_score": baseline.get("score"),
            "candidate_score": candidate.get("score"),
            "improvement": (candidate.get("score", 0.0) - baseline.get("score", 0.0)),
            "promotion_eligible": candidate.get("promotion_eligible", False),
            "critical_pass_rate": candidate.get("critical_pass_rate"),
            "critical_failures": candidate.get("critical_failures", []),
            "promotion_blockers": candidate.get("promotion_blockers", []),
            "components": {
                key: candidate.get(key)
                for key in (
                    "semantic_component", "language_component", "conversation_component",
                    "comprehension_component", "instruction_component",
                    "epistemic_dialogue_component", "binding_component",
                    "arithmetic_component", "robustness_component",
                    "contrastive_component", "coherence_component",
                    "repetition_component", "epistemic_engine_component",
                )
            },
        }
        reports = ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        text = json.dumps(compact, indent=2, ensure_ascii=False)
        (reports / "latest-evaluation.json").write_text(text, encoding="utf-8")
        (reports / f"v{target}-evaluation.json").write_text(text, encoding="utf-8")
    print(f"Final result for v{target}: {'PROMOTED' if result else 'REJECTED'}")


def main():
    parser = argparse.ArgumentParser(prog="python -m butterfly.deliberate")
    parser.add_argument("command", choices=("prepare", "build", "train", "evaluate"))
    args = parser.parse_args()
    recipe = load_recipe()
    {
        "prepare": command_prepare,
        "build": command_build,
        "train": command_train,
        "evaluate": command_evaluate,
    }[args.command](recipe)


if __name__ == "__main__":
    main()
