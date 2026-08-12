from butterfly.learning.dynamic_exam import (
    GENERATORS,
    fresh_pair,
    generate_bank,
    surface_overlap,
    validate_bank,
)
from butterfly.learning.evaluator import benchmark_surface_prompts, normalize_surface
from butterfly.learning.study_exam import study_surface_prompts


def test_dynamic_exam_selection_transfer_are_disjoint():
    for index, family in enumerate(sorted(GENERATORS), 1):
        selection, transfer = fresh_pair(family, 50000 + index, count=6)
        assert not surface_overlap(selection, transfer)


def test_dynamic_exam_different_seeds_change_fingerprint():
    first = generate_bank("two_steps", 111, count=8, mode="selection")
    second = generate_bank("two_steps", 222, count=8, mode="selection")
    assert first["fingerprint"] != second["fingerprint"]


def test_dynamic_exam_never_reuses_fixed_held_out_surfaces():
    held_out = set(benchmark_surface_prompts())
    held_out |= {normalize_surface(p) for p in study_surface_prompts()}
    for index, family in enumerate(sorted(GENERATORS), 1):
        bank = generate_bank(family, 70000 + index, count=8, mode="selection")
        validate_bank(bank)
        surfaces = {normalize_surface(row["prompt"]) for row in bank["cases"]}
        assert not (surfaces & held_out)


def test_two_steps_contains_structural_validator():
    bank = generate_bank("two_steps", 333, count=5, mode="selection")
    assert all(row.get("expected_items") == 2 for row in bank["cases"])
    assert all(len(row.get("ordered_groups") or []) == 2 for row in bank["cases"])
