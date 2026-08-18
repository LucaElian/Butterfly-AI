import io
import os
from pathlib import Path

import butterfly.learning.autonomy as autonomy


def _touch(path, timestamp):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (timestamp, timestamp))


def test_autonomy_runtime_retention_prunes_old_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "ROOT", tmp_path)

    for i in range(5):
        _touch(tmp_path / "logs" / f"autonomy-20260812-00000{i}.log", 100 + i)
    for i in range(5):
        _touch(tmp_path / "benchmarks" / f"comparison-{i}.json", 150 + i)
    for i in range(8):
        _touch(tmp_path / "reports" / f"brain-0.000{i}-training.json", 200 + i)
        _touch(tmp_path / "reports" / f"brain-0.000{i}-evaluation.json", 300 + i)
        _touch(tmp_path / "reports" / f"study-profile-{i}.json", 400 + i)
        _touch(tmp_path / "reports" / "lifelong" / f"diag-{i}.json", 500 + i)
    _touch(tmp_path / "reports" / "latest-training.json", 50)

    removed = autonomy.prune_runtime_outputs(log_keep=3, report_keep=6, benchmark_keep=3)

    assert len(list((tmp_path / "logs").glob("autonomy-*.log"))) == 3
    assert len(list((tmp_path / "reports").glob("brain-*-training.json"))) == 6
    assert len(list((tmp_path / "reports").glob("brain-*-evaluation.json"))) == 6
    assert len(list((tmp_path / "reports").glob("study-profile-*.json"))) == 6
    assert len(list((tmp_path / "reports" / "lifelong").glob("*.json"))) == 6
    assert len(list((tmp_path / "benchmarks").glob("comparison-*.json"))) == 3
    assert (tmp_path / "reports" / "latest-training.json").exists()
    assert set(removed) == {
        "logs",
        "training_reports",
        "evaluation_reports",
        "study_profiles",
        "lifelong_reports",
        "benchmark_comparisons",
    }


def test_autonomy_error_finalization_reuses_initial_snapshots(monkeypatch):
    first_snapshot = {"seed_version": "brain-test", "seed_slot": "lab", "score": 0.0}
    first_lifelong = {"diagnostic": "initial"}

    def fail_if_recomputed(*_args, **_kwargs):
        raise PermissionError("runtime path unavailable")

    monkeypatch.setattr(autonomy, "capability_snapshot", fail_if_recomputed)
    monkeypatch.setattr(autonomy, "strict_dynamic_diagnostic", fail_if_recomputed)

    final_snapshot, final_lifelong = autonomy._final_session_snapshots(
        "error", first_snapshot, first_lifelong
    )

    assert final_snapshot is first_snapshot
    assert final_lifelong is first_lifelong


def test_autonomy_non_error_finalization_refreshes_snapshots(monkeypatch):
    monkeypatch.setattr(autonomy, "capability_snapshot", lambda **_kwargs: {"seed_version": "fresh"})
    monkeypatch.setattr(autonomy, "strict_dynamic_diagnostic", lambda **_kwargs: {"diagnostic": "fresh"})

    for stop_reason in ["completed", "stop_file"]:
        final_snapshot, final_lifelong = autonomy._final_session_snapshots(
            stop_reason, {"seed_version": "initial"}, {"diagnostic": "initial"}
        )

        assert final_snapshot == {"seed_version": "fresh"}
        assert final_lifelong == {"diagnostic": "fresh"}

def test_compact_console_filters_visual_noise_but_log_keeps_full_output():
    console = io.StringIO()
    log = io.StringIO()
    tee = autonomy._Tee(autonomy._CompactConsole(console), log)

    tee.write("ButterflyAI deliberate preflight\n")
    tee.write("comprehension epoch 1/3 |     1/400 | answer-loss 0.9000 | ETA 4.0m\n")
    tee.write("comprehension epoch 1: answer-train=0.8000 answer-valid=0.9000\n")
    tee.write("BLOCK 1 result: rejected score=0.9\n")
    tee.flush()

    visual = console.getvalue()
    full = log.getvalue()

    assert "ButterflyAI deliberate preflight" not in visual
    assert "answer-loss" in visual
    assert "BLOCK 1 result" in visual
    assert "ButterflyAI deliberate preflight" in full
    assert "answer-loss" in full


def test_autonomy_console_stream_defaults_to_full_and_allows_compact(monkeypatch):
    raw = io.StringIO()
    monkeypatch.delenv("BUTTERFLY_AUTONOMY_CONSOLE", raising=False)
    assert autonomy._autonomy_console_stream(raw) is raw

    monkeypatch.setenv("BUTTERFLY_AUTONOMY_CONSOLE", "compact")
    assert isinstance(autonomy._autonomy_console_stream(raw), autonomy._CompactConsole)

    monkeypatch.setenv("BUTTERFLY_AUTONOMY_CONSOLE", "live")
    live_stream = autonomy._autonomy_console_stream(raw)
    assert isinstance(live_stream, autonomy._CompactConsole)
    assert live_stream.live is True


def test_live_console_keeps_stage_context_without_full_noise():
    console = io.StringIO()
    view = autonomy._CompactConsole(console, live=True)

    view.write("ButterflyAI deliberate preflight\n")
    view.write("=== STAGE COMPREHENSION_REPAIR_V2 ===\n")
    view.write("train rows: 123 | valid rows: 45\n")
    view.write("Objective: noisy repeated training policy\n")
    view.write("Recovery: noisy repeated autosave policy\n")
    view.write("comprehension epoch 1/3 |     1/400 | answer-loss 0.9000 | ETA 4.0m\n")
    view.write("comprehension epoch 1: answer-train=0.8000 answer-valid=0.9000\n")
    view.flush()

    visual = console.getvalue()

    assert "ButterflyAI deliberate preflight" not in visual
    assert "=== STAGE" in visual
    assert "train rows:" in visual
    assert "Objective:" not in visual
    assert "Recovery:" not in visual
    assert "answer-loss" in visual

def test_live_console_clears_tty_on_major_phase(monkeypatch):
    class TtyBuffer(io.StringIO):
        def isatty(self):
            return True

    screen = TtyBuffer()
    clear_calls = []
    monkeypatch.setattr(autonomy.os, "system", lambda command: clear_calls.append(command) or 0)

    view = autonomy._CompactConsole(screen, live=True)
    view.write("BLOCK 1/unlimited: instruction_format/two_steps -> autonomy_instruction\n")
    view.write("ButterflyAI deliberate preflight\n")
    view.flush()

    if autonomy.os.name == "nt":
        assert clear_calls == ["cls"]
    else:
        assert "\033[2J\033[H" in screen.getvalue()
    assert "BLOCK 1/unlimited" in screen.getvalue()
    assert "ButterflyAI deliberate preflight" not in screen.getvalue()

def test_runtime_retention_removes_stale_model_artifacts_but_keeps_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "ROOT", tmp_path)
    keep = tmp_path / "models" / "butterfly-vkeep.safetensors"
    stale = tmp_path / "models" / "butterfly-vstale-candidate.safetensors"
    _touch(keep, 100)
    _touch(Path(str(keep) + ".json"), 100)
    _touch(stale, 101)
    _touch(Path(str(stale) + ".json"), 101)
    monkeypatch.setattr(
        autonomy,
        "_registered_model_artifact_paths",
        lambda: {keep.resolve(), Path(str(keep.resolve()) + ".json")},
    )

    removed = autonomy.prune_runtime_outputs(
        log_keep=99,
        report_keep=99,
        benchmark_keep=99,
        prune_stale_model_artifacts=True,
        prune_stale_training_state=False,
        prune_dev_caches=False,
        prune_python_caches=False,
    )

    assert keep.exists()
    assert Path(str(keep) + ".json").exists()
    assert not stale.exists()
    assert not Path(str(stale) + ".json").exists()
    assert removed == {
        "stale_model_artifacts": [
            "models/butterfly-vstale-candidate.safetensors",
            "models/butterfly-vstale-candidate.safetensors.json",
        ]
    }


def test_runtime_retention_removes_training_state_only_when_not_recoverable(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "ROOT", tmp_path)
    resume = tmp_path / "training_state" / "deliberate" / "resume.safetensors"
    progress = tmp_path / "training_state" / "deliberate" / "progress.json"
    _touch(resume, 100)
    _touch(progress, 101)
    monkeypatch.setattr(autonomy, "load_current_experiment", lambda: {"status": "rejected"})

    removed = autonomy.prune_runtime_outputs(
        log_keep=99,
        report_keep=99,
        benchmark_keep=99,
        prune_stale_model_artifacts=False,
        prune_stale_training_state=True,
        prune_dev_caches=False,
        prune_python_caches=False,
    )

    assert not resume.exists()
    assert not progress.exists()
    assert set(removed) == {"stale_training_state"}

    _touch(resume, 100)
    _touch(progress, 101)
    monkeypatch.setattr(autonomy, "load_current_experiment", lambda: {"status": "dataset_ready"})
    removed = autonomy.prune_runtime_outputs(
        log_keep=99,
        report_keep=99,
        benchmark_keep=99,
        prune_stale_model_artifacts=False,
        prune_stale_training_state=True,
        prune_dev_caches=False,
        prune_python_caches=False,
    )

    assert resume.exists()
    assert progress.exists()
    assert removed == {}


def test_runtime_retention_cleans_dev_and_python_caches_without_touching_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "ROOT", tmp_path)
    pytest_cache = tmp_path / ".pytest_cache" / "v" / "cache" / "nodeids"
    pycache = tmp_path / "butterfly" / "__pycache__" / "module.pyc"
    venv_cache = tmp_path / ".venv" / "lib" / "__pycache__" / "keep.pyc"
    _touch(pytest_cache, 100)
    _touch(pycache, 101)
    _touch(venv_cache, 102)

    removed = autonomy.prune_runtime_outputs(
        log_keep=99,
        report_keep=99,
        benchmark_keep=99,
        prune_stale_model_artifacts=False,
        prune_stale_training_state=False,
        prune_dev_caches=True,
        prune_python_caches=True,
    )

    assert not pytest_cache.exists()
    assert not pycache.exists()
    assert venv_cache.exists()
    assert set(removed) == {"dev_caches", "python_caches"}

def test_runtime_retention_keeps_referenced_benchmark_comparisons(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "ROOT", tmp_path)
    referenced = tmp_path / "benchmarks" / "comparison-referenced.json"
    _touch(referenced, 1)
    for i in range(5):
        _touch(tmp_path / "benchmarks" / f"comparison-temp-{i}.json", 100 + i)
    monkeypatch.setattr(autonomy, "_referenced_benchmark_paths", lambda: {referenced.resolve()})

    removed = autonomy.prune_runtime_outputs(
        log_keep=99,
        report_keep=99,
        benchmark_keep=2,
        prune_stale_model_artifacts=False,
        prune_stale_training_state=False,
        prune_dev_caches=False,
        prune_python_caches=False,
    )

    assert referenced.exists()
    assert len(list((tmp_path / "benchmarks").glob("comparison-temp-*.json"))) == 2
    assert set(removed) == {"benchmark_comparisons"}


def test_runtime_retention_removes_unreferenced_legacy_migration_benchmarks(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "ROOT", tmp_path)
    migration = tmp_path / "benchmarks" / "migration-legacy.json"
    referenced = tmp_path / "benchmarks" / "migration-referenced.json"
    _touch(migration, 100)
    _touch(referenced, 101)
    monkeypatch.setattr(autonomy, "_referenced_benchmark_paths", lambda: {referenced.resolve()})

    removed = autonomy.prune_runtime_outputs(
        log_keep=99,
        report_keep=99,
        benchmark_keep=99,
        prune_stale_model_artifacts=False,
        prune_stale_training_state=False,
        prune_dev_caches=False,
        prune_python_caches=False,
    )

    assert not migration.exists()
    assert referenced.exists()
    assert removed == {"legacy_migration_benchmarks": ["benchmarks/migration-legacy.json"]}


def test_runtime_retention_policy_defaults_are_cloud_light():
    policy = autonomy._retention_policy({"runtime_retention": {}})

    assert policy["reports"] == 6
    assert policy["benchmark_comparisons"] == 15

    policy = autonomy._retention_policy({"runtime_retention": {"reports": 3, "benchmark_comparisons": 15}})
    assert policy["reports"] == 3
    assert policy["benchmark_comparisons"] == 15
