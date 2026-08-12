from butterfly import storage


def test_storage_status_summarizes_history_by_default(monkeypatch):
    monkeypatch.setattr(storage, "get_active_entry", lambda: None)
    monkeypatch.setattr(storage, "get_lab_entry", lambda: None)
    monkeypatch.setattr(storage, "get_candidate_entry", lambda: None)
    monkeypatch.setattr(storage, "load_history", lambda: {
        "format": 1,
        "versions": [
            {"version": "a", "status": "rejected"},
            {"version": "b", "status": "lab_accepted"},
            {"version": "c", "status": "rejected"},
        ],
    })

    status = storage.storage_status(history_limit=2)

    assert status["history"]["total_entries"] == 3
    assert status["history"]["status_counts"]["rejected"] == 2
    assert [row["version"] for row in status["history"]["recent"]] == ["b", "c"]
    assert all("metadata" not in row for row in status["history"]["recent"])


def test_storage_status_can_return_full_history(monkeypatch):
    history = {"format": 1, "versions": [{"version": "a", "status": "rejected"}]}
    monkeypatch.setattr(storage, "get_active_entry", lambda: None)
    monkeypatch.setattr(storage, "get_lab_entry", lambda: None)
    monkeypatch.setattr(storage, "get_candidate_entry", lambda: None)
    monkeypatch.setattr(storage, "load_history", lambda: history)

    assert storage.storage_status(full_history=True)["history"] is history
