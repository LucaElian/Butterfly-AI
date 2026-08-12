from butterfly.learning.autonomy import _resource_check


def test_time_budget_stops_session(monkeypatch):
    cfg = {"min_free_disk_gb": 0, "stop_file": ".butterfly/DO_NOT_EXIST"}
    monkeypatch.setattr("butterfly.learning.autonomy.time.monotonic", lambda: 1000.0)
    ok, reason = _resource_check(cfg, started_monotonic=0.0, max_minutes=1.0)
    assert not ok
    assert reason == "time_budget"
