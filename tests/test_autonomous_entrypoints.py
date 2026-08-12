from pathlib import Path


def test_manual_pipeline_entrypoints_are_retired():
    root = Path(__file__).resolve().parents[1]
    retired = {
        "01_PREPARE.bat",
        "02_BUILD_DATASET.bat",
        "03_TRAIN.bat",
        "04_EVALUATE_AND_PROMOTE.bat",
        "RUN_PIPELINE.bat",
        "SLEEP_AND_LEARN.bat",
    }
    for name in retired:
        assert not (root / name).exists()

    assert (root / "RUN_NIGHT_STUDY.bat").exists()
    assert (root / "STOP_NIGHT_STUDY.bat").exists()
    assert (root / "LIFELONG_STATUS.bat").exists()


def test_autonomy_config_replaces_manual_pipeline_config():
    root = Path(__file__).resolve().parents[1]
    assert (root / "config" / "autonomy.json").exists()
    assert not (root / "config" / "pipeline.json").exists()
    assert not (root / "butterfly" / "pipeline.py").exists()
