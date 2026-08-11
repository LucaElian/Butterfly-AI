import butterfly.pipeline as pipeline


def _state(status="pending"):
    return {
        "schema_version": pipeline.STATE_SCHEMA,
        "experiment_id": "example",
        "stages": {
            name: {
                "status": status,
                "signature": None,
                "completed_at": None,
                "log": None,
            }
            for name in pipeline.STAGE_NAMES
        },
        "last_error": "old error",
    }


def test_terminal_experiment_detection():
    for status in ("promoted", "lab_accepted", "rejected", "cancelled"):
        assert pipeline._is_terminal_experiment({"status": status})
    assert not pipeline._is_terminal_experiment({"status": "candidate_ready"})


def test_terminal_state_is_repaired_to_complete(monkeypatch):
    state = _state()
    exp = {"status": "lab_accepted", "updated_at": 123.0}
    monkeypatch.setattr(pipeline, "save_state", lambda value: None)

    pipeline._lock_terminal_state(state, exp)

    assert all(
        state["stages"][name]["status"] == "complete"
        for name in pipeline.STAGE_NAMES
    )
    assert all(
        state["stages"][name]["completed_at"] == 123.0
        for name in pipeline.STAGE_NAMES
    )
    assert state["last_error"] is None


def test_terminal_refresh_never_recomputes_signatures(monkeypatch):
    state = _state(status="complete")
    exp = {"status": "lab_accepted", "updated_at": 123.0}

    monkeypatch.setattr(pipeline, "save_state", lambda value: None)
    monkeypatch.setattr(
        pipeline,
        "_stage_signature",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("terminal signatures must not be recomputed")
        ),
    )

    pipeline._refresh_validity(state, {}, exp, {})
    assert all(
        state["stages"][name]["status"] == "complete"
        for name in pipeline.STAGE_NAMES
    )
