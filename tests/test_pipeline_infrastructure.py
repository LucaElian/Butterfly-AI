import json
from pathlib import Path

def test_permanent_pipeline_recipe_exists_and_is_idle_before_v00052():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "pipeline.json").read_text(encoding="utf-8"))
    assert cfg["schema_version"] == 1
    assert cfg["target_brain"] is None
    assert cfg["benchmark_suite"] == "0.00042"
    assert set(cfg["stages"]) == {
        "prepare", "build_dataset", "train", "evaluate_and_promote"
    }

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
