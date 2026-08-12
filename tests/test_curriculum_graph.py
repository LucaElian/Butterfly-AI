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
