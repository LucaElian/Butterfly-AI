def test_registry_api_has_explicit_slots():
    import butterfly.registry as registry
    for name in (
        "get_active_entry",
        "get_lab_entry",
        "get_candidate_entry",
        "get_seed_entry",
        "promote_to_lab",
        "promote_to_active",
        "compact_physical_models",
    ):
        assert hasattr(registry, name)


def test_seed_prefers_lab(monkeypatch):
    import butterfly.registry as registry
    monkeypatch.setattr(registry, "get_lab_entry", lambda: {"version": "lab"})
    monkeypatch.setattr(registry, "get_active_entry", lambda: {"version": "active"})
    assert registry.get_seed_entry()["version"] == "lab"
