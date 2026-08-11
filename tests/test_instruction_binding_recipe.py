import json

from butterfly.config import ROOT
from butterfly.corpus.skills import BUILDERS
from butterfly.learning.evaluator import benchmark_surface_prompts, normalize_surface


def test_instruction_following_builder_is_registered():
    assert "instruction_following" in BUILDERS


def test_instruction_following_train_valid_are_disjoint_and_held_out():
    train, valid = BUILDERS["instruction_following"](123456)
    assert train
    assert valid

    train_families = {row["family"] for row in train}
    valid_families = {row["family"] for row in valid}
    assert train_families.isdisjoint(valid_families)

    held_out = benchmark_surface_prompts()
    assert all(normalize_surface(row["user"]) not in held_out for row in train + valid)


def test_instruction_binding_recipe_is_generic_and_uses_known_skills():
    path = ROOT / "config" / "recipes" / "instruction_binding.json"
    recipe = json.loads(path.read_text(encoding="utf-8"))

    assert recipe["schema_version"] == 1
    assert recipe["name"] == "instruction_binding"
    assert recipe["focus_metrics"] == ["binding_component", "instruction_format_component"]
    assert "target_brain" not in recipe
    assert "expected_active" not in recipe
    assert "benchmark_suite" not in recipe

    for stage in recipe["training_stages"]:
        for skill in stage["skills"]:
            assert skill["name"] in BUILDERS


def test_instruction_binding_recipe_protects_retained_lab_capabilities():
    recipe = json.loads(
        (ROOT / "config" / "recipes" / "instruction_binding.json").read_text(encoding="utf-8")
    )
    protected = set(recipe["protected_metrics"])
    assert {
        "intent_routing_component",
        "conversation_component",
        "comprehension_component",
        "epistemic_dialogue_component",
        "language_component",
        "coherence_component",
        "repetition_component",
    }.issubset(protected)


def test_instruction_binding_stages_use_held_out_study_policy():
    recipe = json.loads((ROOT / "config" / "recipes" / "instruction_binding.json").read_text(encoding="utf-8"))
    for stage in recipe["training_stages"]:
        assert stage["study_focus_metrics"] == ["binding_exact_component", "instruction_format_component"]
        assert {"retention_conversation_component", "retention_comprehension_component", "retention_epistemic_component", "retention_intent_component", "retention_quality_component"}.issubset(set(stage["study_protected_metrics"]))
