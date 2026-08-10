import json
from pathlib import Path

from butterfly.learning.evaluator import BENCHMARK_SUITE_VERSION


def test_permanent_pipeline_is_idle_between_deliberate_candidates():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(
        (root / "config" / "pipeline.json").read_text(encoding="utf-8")
    )

    assert cfg["schema_version"] == 1
    assert cfg["infrastructure_version"] == "1.0"

    # Between rejected/accepted deliberate candidates, pipeline must be idle.
    # A future candidate intentionally sets a target again.
    assert cfg["target_brain"] is None

    # Permanent invariant: pipeline and evaluator must agree.
    # Do NOT hardcode a suite number in this test.
    assert cfg["benchmark_suite"] == BENCHMARK_SUITE_VERSION

    assert set(cfg["stages"]) == {
        "prepare",
        "build_dataset",
        "train",
        "evaluate_and_promote",
    }
    assert all(
        isinstance(value, list) and value
        for value in cfg["stages"].values()
    )


def test_pipeline_names_are_version_agnostic():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "01_PREPARE.bat",
        "02_BUILD_DATASET.bat",
        "03_TRAIN.bat",
        "04_EVALUATE_AND_PROMOTE.bat",
        "RUN_PIPELINE.bat",
    ):
        assert (root / name).exists()
