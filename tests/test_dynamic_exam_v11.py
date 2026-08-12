from butterfly.learning.dynamic_exam import (
    GENERATORS,
    fresh_pair,
    surface_overlap,
)


def test_every_dynamic_family_builds_large_disjoint_selection_transfer_pair():
    for index, family in enumerate(sorted(GENERATORS), 1):
        selection, transfer = fresh_pair(family, 880000 + index, count=16)
        assert not surface_overlap(selection, transfer)
        assert selection["fingerprint"] != transfer["fingerprint"]
        assert len(selection["cases"]) == 16
        assert len(transfer["cases"]) == 16


def test_fresh_pair_is_deterministic_for_same_seed():
    first_s, first_t = fresh_pair("api", 990001, count=12)
    second_s, second_t = fresh_pair("api", 990001, count=12)
    assert first_s["fingerprint"] == second_s["fingerprint"]
    assert first_t["fingerprint"] == second_t["fingerprint"]
