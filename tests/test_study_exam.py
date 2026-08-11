from butterfly.corpus.skills.binding import build as build_binding
from butterfly.learning.evaluator import CASES, PROMOTION_THRESHOLDS, normalize_surface
from butterfly.learning.study_exam import STUDY_CASES, STUDY_RESERVED_EXACT_TARGETS, study_surface_prompts

def test_study_binding_targets_are_not_training_answers():
    train, valid = build_binding(73129)
    answers = {row["assistant"].casefold() for row in train + valid}
    assert all(x.casefold() not in answers for x in STUDY_RESERVED_EXACT_TARGETS)

def test_study_surfaces_are_distinct_from_final_benchmark():
    final = {normalize_surface(x["prompt"]) for x in CASES}
    study = {normalize_surface(x) for x in study_surface_prompts()}
    assert final.isdisjoint(study)

def test_study_exam_has_focus_and_separate_retention_groups():
    groups = {x["group"] for x in STUDY_CASES}
    assert {
        "binding", "instruction_format", "retention_conversation",
        "retention_comprehension", "retention_epistemic"
    }.issubset(groups)

def test_promotion_threshold_splits_instruction_from_arithmetic():
    assert "instruction_format_component" in PROMOTION_THRESHOLDS
    assert "instruction_component" not in PROMOTION_THRESHOLDS
    assert "arithmetic_component" in PROMOTION_THRESHOLDS

def test_final_instruction_format_excludes_exact_skills():
    rows = [x for x in CASES if x["category"] == "instruction" and x.get("skill") not in {"binding", "arithmetic"}]
    assert rows
