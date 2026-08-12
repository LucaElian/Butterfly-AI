from pathlib import Path
import ast


def _top_level_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }


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

    try:
        parser.parse_args(["sleep"])
    except SystemExit:
        pass
    else:
        raise AssertionError("sleep should no longer be a public CLI command")


def test_cli_imports_are_lightweight_for_diagnostics():
    root = Path(__file__).resolve().parents[1]
    top_level_imports = _top_level_imports(root / "butterfly" / "cli.py")
    assert "training.runtime" not in top_level_imports
    assert "checkpoint" not in top_level_imports
    assert "learning.sleep_cycle" not in top_level_imports


def test_storage_status_does_not_import_checkpoint_runtime():
    root = Path(__file__).resolve().parents[1]
    top_level_imports = _top_level_imports(root / "butterfly" / "storage.py")
    assert "checkpoint" not in top_level_imports


def test_evaluator_runtime_imports_are_lightweight():
    root = Path(__file__).resolve().parents[1]
    evaluator_imports = _top_level_imports(root / "butterfly" / "learning" / "evaluator.py")
    runtime_imports = _top_level_imports(root / "butterfly" / "runtime.py")
    engine_imports = _top_level_imports(root / "butterfly" / "epistemic" / "engine.py")

    assert "generation" not in runtime_imports
    assert "web" not in engine_imports
    assert "generation" not in evaluator_imports


def test_legacy_learning_modules_are_retired():
    root = Path(__file__).resolve().parents[1]
    retired = {
        "butterfly/learning/sleep_cycle.py",
        "butterfly/agent/postflight.py",
        "butterfly/corpus/builder.py",
        "butterfly/trainer.py",
        "config/recipes/sleep_learning.json",
    }
    for rel in retired:
        assert not (root / rel).exists()
