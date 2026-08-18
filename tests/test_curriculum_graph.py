from butterfly.learning.curriculum_graph import (
    load_seed,
    seed_fingerprint,
    validate_seed,
)


def test_curriculum_seed_is_valid_and_acyclic():
    seed = load_seed()
    validate_seed(seed)
    assert len(seed["nodes"]) >= 70


def test_curriculum_ids_are_unique():
    seed = load_seed()
    ids = [row["id"] for row in seed["nodes"]]
    assert len(ids) == len(set(ids))


def test_curriculum_spans_multiple_domains():
    seed = load_seed()
    domains = {row["domain"] for row in seed["nodes"]}
    assert {"computing", "programming", "windows", "web", "reasoning", "security", "learning"} <= domains


def test_curriculum_fingerprint_is_stable():
    seed = load_seed()
    assert seed_fingerprint(seed) == seed_fingerprint(seed)


def test_verified_material_reopens_plateau_node(tmp_path, monkeypatch):
    import butterfly.learning.curriculum_graph as cg

    monkeypatch.setattr(cg, "STATE_PATH", tmp_path / "curriculum_graph.json")
    state = cg.bootstrap_graph()
    node_id = "instruction.two_steps"
    state["nodes"][node_id]["plateau"] = True
    state["nodes"][node_id]["status"] = "plateau"
    state["nodes"][node_id]["strategy_failures"] = {"balanced": 1, "focused": 1, "gentle": 1}
    cg.save_json(cg.STATE_PATH, state)

    cg.mark_material(node_id, "verified_packet")
    updated, _ = cg.load_graph()
    row = updated["nodes"][node_id]
    assert row["material_status"] == "verified_packet"
    assert row["plateau"] is False
    assert row["strategy_failures"] == {}
    assert row["status"] == "available"