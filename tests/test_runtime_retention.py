import os

import butterfly.learning.night_study as night_study


def _touch(path, timestamp):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (timestamp, timestamp))


def test_night_study_runtime_retention_prunes_old_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(night_study, "ROOT", tmp_path)

    for i in range(5):
        _touch(tmp_path / "logs" / f"night-study-20260812-00000{i}.log", 100 + i)
    for i in range(8):
        _touch(tmp_path / "reports" / f"brain-0.000{i}-training.json", 200 + i)
        _touch(tmp_path / "reports" / f"brain-0.000{i}-evaluation.json", 300 + i)
        _touch(tmp_path / "reports" / f"study-profile-{i}.json", 400 + i)
        _touch(tmp_path / "reports" / "lifelong" / f"diag-{i}.json", 500 + i)
    _touch(tmp_path / "reports" / "latest-training.json", 50)

    removed = night_study.prune_runtime_outputs(log_keep=3, report_keep=6)

    assert len(list((tmp_path / "logs").glob("night-study-*.log"))) == 3
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
