import os

import butterfly.learning.autonomy as autonomy


def _touch(path, timestamp):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (timestamp, timestamp))


def test_autonomy_runtime_retention_prunes_old_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomy, "ROOT", tmp_path)

    for i in range(5):
        _touch(tmp_path / "logs" / f"autonomy-20260812-00000{i}.log", 100 + i)
    for i in range(8):
        _touch(tmp_path / "reports" / f"brain-0.000{i}-training.json", 200 + i)
        _touch(tmp_path / "reports" / f"brain-0.000{i}-evaluation.json", 300 + i)
        _touch(tmp_path / "reports" / f"study-profile-{i}.json", 400 + i)
        _touch(tmp_path / "reports" / "lifelong" / f"diag-{i}.json", 500 + i)
    _touch(tmp_path / "reports" / "latest-training.json", 50)

    removed = autonomy.prune_runtime_outputs(log_keep=3, report_keep=6)

    assert len(list((tmp_path / "logs").glob("autonomy-*.log"))) == 3
    assert len(list((tmp_path / "reports").glob("brain-*-training.json"))) == 6
    assert len(list((tmp_path / "reports").glob("brain-*-evaluation.json"))) == 6
    assert len(list((tmp_path / "reports").glob("study-profile-*.json"))) == 6
    assert len(list((tmp_path / "reports" / "lifelong").glob("*.json"))) == 6
    assert (tmp_path / "reports" / "latest-training.json").exists()
    assert set(removed) == {
        "logs",
        "training_reports",
        "evaluation_reports",
        "study_profiles",
        "lifelong_reports",
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


def test_autonomy_completed_finalization_refreshes_snapshots(monkeypatch):
    monkeypatch.setattr(autonomy, "capability_snapshot", lambda **_kwargs: {"seed_version": "fresh"})
    monkeypatch.setattr(autonomy, "strict_dynamic_diagnostic", lambda **_kwargs: {"diagnostic": "fresh"})

    final_snapshot, final_lifelong = autonomy._final_session_snapshots(
        "completed", {"seed_version": "initial"}, {"diagnostic": "initial"}
    )

    assert final_snapshot == {"seed_version": "fresh"}
    assert final_lifelong == {"diagnostic": "fresh"}
