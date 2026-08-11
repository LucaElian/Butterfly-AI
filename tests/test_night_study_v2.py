import json

from butterfly.config import ROOT
from butterfly.learning.evaluator import benchmark_surface_prompts, normalize_surface
from butterfly.learning.night_study import choose_lesson
from butterfly.learning.study_exam import STUDY_CASES
from butterfly.upgrade import _lab_focus_check


def test_study_surfaces_remain_disjoint_from_final_benchmark():
    final = benchmark_surface_prompts()
    study = {normalize_surface(case["prompt"]) for case in STUDY_CASES}
    assert not (final & study)


def test_instruction_shadow_has_balanced_families():
    fmt = [c for c in STUDY_CASES if c.get("group") == "instruction_format"]
    counts = {}
    for case in fmt:
        counts[case["format_family"]] = counts.get(case["format_family"], 0) + 1
    assert set(counts) == {"sentence", "two_steps", "missing", "short"}
    assert min(counts.values()) >= 5


def test_night_curriculum_can_prioritize_critical_with_zero_metric_gap():
    snapshot = {
        "capabilities": [
            {
                "capability": "conversation",
                "gap": 0.0,
                "critical_count": 1,
                "priority": 0.12,
                "trainable": True,
                "recipe": "night_conversation",
            },
            {
                "capability": "instruction_format",
                "gap": 0.02,
                "critical_count": 0,
                "priority": 0.02,
                "trainable": True,
                "recipe": "night_instruction",
            },
        ]
    }
    assert choose_lesson(snapshot)["capability"] == "conversation"


def test_lab_can_accept_hard_gate_closure():
    seed = {
        "instruction_format_component": 0.695,
        "critical_failures": [],
        "cases": [],
    }
    candidate = {
        "instruction_format_component": 0.730,
        "critical_failures": [],
        "cases": [],
    }
    recipe = {
        "focus_metrics": ["instruction_format_component"],
        "protected_metrics": [],
        "lab_acceptance": {"allow_hard_gate_closure": True},
    }
    policy = {
        "lab": {
            "min_focus_delta": 0.05,
            "max_protected_regression": 0.03,
            "require_all_focus_metrics": True,
        }
    }
    ok, failures = _lab_focus_check(candidate, seed, recipe, policy)
    assert ok, failures


def test_lab_can_accept_clean_critical_repair_near_ceiling():
    seed = {
        "conversation_component": 0.9635,
        "critical_failures": ["hello_new"],
        "cases": [{"id": "hello_new", "category": "conversation"}],
    }
    candidate = {
        "conversation_component": 0.9600,
        "critical_failures": [],
        "cases": [{"id": "hello_new", "category": "conversation"}],
    }
    recipe = {
        "focus_metrics": ["conversation_component"],
        "protected_metrics": [],
        "lab_acceptance": {
            "allow_critical_repair": True,
            "critical_categories": ["conversation"],
            "max_focus_regression": 0.01,
        },
    }
    policy = {
        "lab": {
            "min_focus_delta": 0.05,
            "max_protected_regression": 0.03,
            "require_all_focus_metrics": True,
        }
    }
    ok, failures = _lab_focus_check(candidate, seed, recipe, policy)
    assert ok, failures


def test_night_config_has_all_remaining_neural_categories():
    cfg = json.loads((ROOT / "config" / "night_study.json").read_text(encoding="utf-8"))
    assert {"conversation", "comprehension", "instruction_format", "epistemic_dialogue"} <= set(cfg["curriculum"])
    assert cfg["max_blocks_per_session"] >= 3
