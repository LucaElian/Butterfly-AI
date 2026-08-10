import json
from pathlib import Path


def test_permanent_pipeline_reuses_same_stage_names_for_v00053():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "pipeline.json").read_text(encoding="utf-8"))
    assert cfg["schema_version"] == 1
    assert cfg["infrastructure_version"] == "1.0"
    assert cfg["target_brain"] == "0.00053"
    assert cfg["benchmark_suite"] == "0.00043"
    assert set(cfg["stages"]) == {"prepare", "build_dataset", "train", "evaluate_and_promote"}
    assert all(isinstance(value, list) and value for value in cfg["stages"].values())


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
