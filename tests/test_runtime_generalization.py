import json
import re
from pathlib import Path

from butterfly.experiments import next_version_from_values
from butterfly.learning.evaluator import BENCHMARK_SUITE_ID


def test_next_brain_version_comes_from_history_values():
    assert next_version_from_values(["1.0004", "1.0051", "1.0053"]) == "1.0054"


def test_evaluator_suite_is_fingerprint_not_manual_brain_style_version():
    assert re.fullmatch(r"suite-[0-9a-f]{12}", BENCHMARK_SUITE_ID)


def test_pipeline_config_contains_behavior_not_runtime_target():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "pipeline.json").read_text(encoding="utf-8"))
    assert cfg["schema_version"] == 2
    assert "target_brain" not in cfg
    assert "benchmark_suite" not in cfg
    assert cfg["default_recipe"]


def test_recipes_contain_no_brain_or_suite_identity():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "config" / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text(encoding="utf-8"))
        assert "target_brain" not in recipe, path
        assert "expected_active" not in recipe, path
        assert "benchmark_suite" not in recipe, path
