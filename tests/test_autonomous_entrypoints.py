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

    assert (root / "RUN_AUTONOMY.bat").exists()
    assert (root / "STOP_AUTONOMY.bat").exists()
    assert (root / "AUTONOMY_STATUS.bat").exists()


def test_autonomy_config_replaces_manual_pipeline_config():
    root = Path(__file__).resolve().parents[1]
    assert (root / "config" / "autonomy.json").exists()
    assert not (root / "config" / "pipeline.json").exists()
    assert not (root / "butterfly" / "pipeline.py").exists()


def test_public_learning_cli_is_autonomy():
    from butterfly.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["autonomy", "--dry-run"])
    assert args.func.__name__ == "command_autonomy"

    retired_command = "ni" + "ght" + "-study"
    try:
        parser.parse_args([retired_command])
    except SystemExit:
        pass
    else:
        raise AssertionError(f"{retired_command} should no longer be a public CLI command")
