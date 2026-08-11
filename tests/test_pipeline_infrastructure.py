import json
from pathlib import Path


def test_pipeline_files_are_permanent_and_config_is_general():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "pipeline.json").read_text(encoding="utf-8"))
    assert cfg["schema_version"] == 2
    assert set(cfg["stages"]) == {
        "prepare",
        "build_dataset",
        "train",
        "evaluate_and_promote",
    }
    assert "target_brain" not in cfg
    assert "benchmark_suite" not in cfg

    for name in (
        "01_PREPARE.bat",
        "02_BUILD_DATASET.bat",
        "03_TRAIN.bat",
        "04_EVALUATE_AND_PROMOTE.bat",
        "RUN_PIPELINE.bat",
    ):
        assert (root / name).exists()
