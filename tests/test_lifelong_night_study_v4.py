import json
from pathlib import Path

from butterfly.config import ROOT
from butterfly.deliberate_trainer import DYNAMIC_FOCUS_KEY, _failure_driven_stage_cfg
from butterfly.learning.curriculum_graph import load_seed
from butterfly.learning.dynamic_exam import (
    GENERATORS,
    fresh_bank_excluding,
    fresh_pair,
    normalize_surface,
)
from butterfly.learning.lifelong_bridge import load_lifelong_config
from butterfly.learning.night_study import load_night_config
from butterfly.upgrade import _lab_focus_check


def test_dynamic_engine_expands_beyond_original_families():
    assert {"dataset", "epoch", "ram", "cpu"} <= set(GENERATORS)
    assert len(GENERATORS) >= 20


def test_acceptance_bank_is_disjoint_from_selection_and_transfer():
    selection, transfer = fresh_pair("api", 470001, count=8)
    forbidden = {
        normalize_surface(row["prompt"])
        for bank in (selection, transfer)
        for row in bank["cases"]
    }
    acceptance = fresh_bank_excluding(
        "api", 470002, count=8, mode="acceptance", forbidden_surfaces=forbidden
    )
    acceptance_surfaces = {normalize_surface(row["prompt"]) for row in acceptance["cases"]}
    assert not (acceptance_surfaces & forbidden)


def test_material_bindings_reference_real_nodes_and_dynamic_families():
    cfg = load_lifelong_config()
    seed = load_seed()
    nodes = {row["id"] for row in seed["nodes"]}
    bindings = cfg["material_bindings"]
    assert len(bindings) >= 17
    for node_id, binding in bindings.items():
        assert node_id in nodes
        assert binding["dynamic_family"] in GENERATORS
        assert binding["recipe"].startswith("night_")
        assert binding["source"] == "internal_verified"


def test_curriculum_now_spans_instruction_and_conversation_subskills():
    seed = load_seed()
    ids = {row["id"] for row in seed["nodes"]}
    assert {
        "instruction.two_steps",
        "instruction.missing",
        "conversation.greeting",
        "conversation.thanks",
        "programming.api_basics",
        "ai.training_epochs",
    } <= ids
    assert len(ids) >= 97


def test_dynamic_stage_uses_fresh_selection_as_primary_focus_and_strategy_lr():
    stage = {
        "name": "demo",
        "lr": 4e-6,
        "study_focus_metrics": ["retention_folder_component"],
        "study_protected_metrics": ["retention_folder_component", "retention_quality_component"],
        "failure_focus_min_delta": 0.02,
    }
    experiment = {
        "focus_target": {
            "dynamic_family": "folder",
            "selection_min_delta": 0.03,
            "lr_scale": 0.5,
        }
    }
    result = _failure_driven_stage_cfg(stage, experiment)
    assert result["study_focus_metrics"] == [DYNAMIC_FOCUS_KEY]
    assert result["min_each_study_focus_delta"] == 0.03
    assert result["lr"] == 2e-6
    # Fixed held-out shadows remain protected while the fresh exam selects checkpoints.
    assert "retention_folder_component" in result["study_protected_metrics"]


def test_lifelong_dynamic_acceptance_can_preserve_real_new_learning_without_fixed_focus_delta():
    candidate_metrics = {
        "comprehension_component": 0.81,
        "critical_failures": ["folder_new"],
    }
    seed_metrics = {
        "comprehension_component": 0.82,
        "critical_failures": ["folder_new"],
    }
    recipe = {
        "focus_metrics": ["comprehension_component"],
        "protected_metrics": [],
    }
    policy = {
        "lab": {
            "min_focus_delta": 0.05,
            "require_all_focus_metrics": True,
            "max_protected_regression": 0.03,
        }
    }
    candidate_entry = {
        "metadata": {
            "focus_target": {
                "lifelong_mode": True,
                "max_fixed_focus_regression": 0.02,
            },
            "lifelong_acceptance": {
                "passed": True,
                "delta": 0.08,
                "candidate_score": 0.84,
            },
        }
    }
    ok, failures = _lab_focus_check(
        candidate_metrics,
        seed_metrics,
        recipe,
        policy,
        candidate_entry=candidate_entry,
    )
    assert ok, failures


def test_lifelong_acceptance_never_allows_new_fixed_critical_failure():
    candidate_metrics = {
        "comprehension_component": 0.81,
        "critical_failures": ["folder_new", "thanks_new"],
    }
    seed_metrics = {
        "comprehension_component": 0.82,
        "critical_failures": ["folder_new"],
    }
    recipe = {"focus_metrics": ["comprehension_component"], "protected_metrics": []}
    policy = {
        "lab": {
            "min_focus_delta": 0.05,
            "require_all_focus_metrics": True,
            "max_protected_regression": 0.03,
        }
    }
    candidate_entry = {
        "metadata": {
            "focus_target": {"lifelong_mode": True, "max_fixed_focus_regression": 0.02},
            "lifelong_acceptance": {"passed": True},
        }
    }
    ok, _ = _lab_focus_check(
        candidate_metrics,
        seed_metrics,
        recipe,
        policy,
        candidate_entry=candidate_entry,
    )
    assert not ok


def test_night_study_zero_budget_means_user_stoppable_unlimited_session():
    cfg = load_night_config()
    assert cfg["engine_version"] == 4
    assert cfg["max_blocks_per_session"] == 0
    assert cfg["max_minutes_per_session"] == 0
    assert cfg["stop_after_active_promotion"] is False
    assert cfg["lifelong_learning"]["research_engine_enabled"] is False


def test_lifelong_mode_requires_fresh_acceptance_even_if_fixed_focus_improves():
    candidate_metrics = {
        "comprehension_component": 0.90,
        "critical_failures": [],
    }
    seed_metrics = {
        "comprehension_component": 0.80,
        "critical_failures": [],
    }
    recipe = {"focus_metrics": ["comprehension_component"], "protected_metrics": []}
    policy = {
        "lab": {
            "min_focus_delta": 0.05,
            "require_all_focus_metrics": True,
            "max_protected_regression": 0.03,
        }
    }
    candidate_entry = {
        "metadata": {
            "focus_target": {"lifelong_mode": True},
            "lifelong_acceptance": {"passed": False},
        }
    }
    ok, failures = _lab_focus_check(
        candidate_metrics,
        seed_metrics,
        recipe,
        policy,
        candidate_entry=candidate_entry,
    )
    assert not ok
    assert any("fresh lifelong acceptance" in failure for failure in failures)
