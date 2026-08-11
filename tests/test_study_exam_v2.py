from butterfly.learning.evaluator import CASES, normalize_surface
from butterfly.learning.study_exam import STUDY_CASES, study_surface_prompts

def test_shadow_exam_is_disjoint_from_final_exam():
    final = {normalize_surface(c["prompt"]) for c in CASES}
    shadow = {normalize_surface(p) for p in study_surface_prompts()}
    assert final.isdisjoint(shadow)

def test_shadow_exam_has_separate_retention_groups():
    groups = {c["group"] for c in STUDY_CASES}
    assert {
        "binding", "instruction_format", "retention_conversation",
        "retention_comprehension", "retention_epistemic"
    }.issubset(groups)
