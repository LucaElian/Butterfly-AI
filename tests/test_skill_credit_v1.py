from butterfly.learning.skill_credit import (
    _best_distillable_attempt,
    _validated_alpha_ladder,
    dynamic_credit_gate,
    fixed_safety_check,
    record_rejected_subject_distillation,
)
from butterfly.memory import MemoryStore


def test_skill_credit_accepts_real_skill_gain():
    ok, failures = dynamic_credit_gate(
        0.76, 0.70, minimum_delta=0.02, minimum_score=0.60
    )
    assert ok, failures


def test_skill_credit_rejects_tiny_gain():
    ok, _ = dynamic_credit_gate(
        0.711, 0.70, minimum_delta=0.02, minimum_score=0.60
    )
    assert not ok


def test_skill_credit_rejects_new_fixed_critical_failure():
    cfg = {
        "max_overall_regression": 0.005,
        "max_component_regression": 0.01,
        "require_no_new_critical": True,
    }
    seed = {
        "score": 0.80,
        "critical_failures": ["existing"],
        "conversation_component": 0.90,
    }
    trial = {
        "score": 0.80,
        "critical_failures": ["existing", "new"],
        "conversation_component": 0.90,
    }
    ok, failures = fixed_safety_check(trial, seed, cfg)
    assert not ok
    assert any("new critical" in x for x in failures)


def test_skill_credit_rejects_other_subject_regression():
    cfg = {
        "max_overall_regression": 0.005,
        "max_component_regression": 0.01,
        "require_no_new_critical": True,
    }
    seed = {
        "score": 0.80,
        "critical_failures": [],
        "conversation_component": 0.90,
        "instruction_format_component": 0.65,
    }
    trial = {
        "score": 0.80,
        "critical_failures": [],
        "conversation_component": 0.87,
        "instruction_format_component": 0.72,
    }
    ok, failures = fixed_safety_check(trial, seed, cfg)
    assert not ok
    assert any("conversation_component" in x for x in failures)


def test_skill_credit_allows_safe_partial_consolidation():
    cfg = {
        "max_overall_regression": 0.005,
        "max_component_regression": 0.01,
        "require_no_new_critical": True,
    }
    seed = {
        "score": 0.80,
        "critical_failures": [],
        "conversation_component": 0.90,
        "instruction_format_component": 0.65,
    }
    trial = {
        "score": 0.799,
        "critical_failures": [],
        "conversation_component": 0.895,
        "instruction_format_component": 0.72,
    }
    ok, failures = fixed_safety_check(trial, seed, cfg)
    assert ok, failures


def test_skill_credit_allows_positive_tradeoff_for_large_verified_gain():
    cfg = {
        "max_overall_regression": 0.005,
        "max_component_regression": 0.01,
        "allow_positive_tradeoffs": True,
        "tradeoff_min_skill_delta": 0.10,
        "tradeoff_max_overall_regression": 0.008,
        "tradeoff_max_component_regression": 0.04,
        "tradeoff_max_regressed_components": 3,
        "require_no_new_critical": True,
    }
    seed = {
        "score": 0.9173,
        "critical_failures": [],
        "conversation_component": 1.0,
        "epistemic_dialogue_component": 0.8450,
        "intent_routing_component": 0.9643,
    }
    trial = {
        "score": 0.9131,
        "critical_failures": [],
        "conversation_component": 0.9635,
        "epistemic_dialogue_component": 0.8140,
        "intent_routing_component": 0.9286,
    }
    ok, failures = fixed_safety_check(trial, seed, cfg, skill_delta=0.1033)
    assert ok, failures


def test_skill_credit_rejects_tradeoff_without_large_verified_gain():
    cfg = {
        "max_overall_regression": 0.005,
        "max_component_regression": 0.01,
        "allow_positive_tradeoffs": True,
        "tradeoff_min_skill_delta": 0.10,
        "tradeoff_max_overall_regression": 0.008,
        "tradeoff_max_component_regression": 0.04,
        "tradeoff_max_regressed_components": 3,
        "require_no_new_critical": True,
    }
    seed = {
        "score": 0.9173,
        "critical_failures": [],
        "conversation_component": 1.0,
    }
    trial = {
        "score": 0.9131,
        "critical_failures": [],
        "conversation_component": 0.9635,
    }
    ok, failures = fixed_safety_check(trial, seed, cfg, skill_delta=0.04)
    assert not ok
    assert failures


def test_skill_credit_rejects_oversized_component_tradeoff():
    cfg = {
        "max_overall_regression": 0.005,
        "max_component_regression": 0.01,
        "allow_positive_tradeoffs": True,
        "tradeoff_min_skill_delta": 0.10,
        "tradeoff_max_overall_regression": 0.008,
        "tradeoff_max_component_regression": 0.04,
        "tradeoff_max_regressed_components": 3,
        "require_no_new_critical": True,
    }
    seed = {
        "score": 0.9173,
        "critical_failures": [],
        "conversation_component": 1.0,
        "intent_routing_component": 0.9643,
    }
    trial = {
        "score": 0.9178,
        "critical_failures": [],
        "conversation_component": 0.9271,
        "intent_routing_component": 0.8929,
    }
    ok, failures = fixed_safety_check(trial, seed, cfg, skill_delta=0.155)
    assert not ok
    assert any("conversation_component" in x for x in failures)

def test_alpha_ladder_is_largest_safe_fraction_first():
    assert _validated_alpha_ladder(
        {"alpha_ladder": [0.35, 1.0, 0.75, 0.5, 0.75]}
    ) == [1.0, 0.75, 0.5, 0.35]


def test_rejected_subject_distillation_selects_large_failed_gain():
    cfg = {
        "minimum_skill_score": 0.6,
        "rejected_subject_distillation": {
            "enabled": True,
            "min_skill_delta": 0.04,
            "min_skill_score": 0.6,
        },
    }
    attempts = [
        {"alpha": 1.0, "skill_score": 0.845, "skill_delta": 0.155, "skill_gate_passed": True, "fixed_gate_passed": False},
        {"alpha": 0.5, "skill_score": 0.69, "skill_delta": 0.0, "skill_gate_passed": False},
    ]
    best = _best_distillable_attempt(attempts, cfg)
    assert best["alpha"] == 1.0


def test_rejected_subject_distillation_ignores_tiny_gain():
    cfg = {
        "minimum_skill_score": 0.6,
        "rejected_subject_distillation": {
            "enabled": True,
            "min_skill_delta": 0.04,
            "min_skill_score": 0.6,
        },
    }
    attempts = [
        {"alpha": 1.0, "skill_score": 0.62, "skill_delta": 0.015, "skill_gate_passed": True},
    ]
    assert _best_distillable_attempt(attempts, cfg) is None


def test_rejected_subject_distillation_stores_verified_packet_once(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    marked = []
    cfg = {
        "rejected_subject_distillation": {
            "enabled": True,
            "quality": 0.88,
        }
    }
    attempt = {
        "alpha": 1.0,
        "skill_score": 0.845,
        "skill_delta": 0.155,
        "fixed_failures": ["conversation_component regressed"],
    }
    first = record_rejected_subject_distillation(
        node_id="computing.files",
        family="file",
        experiment_id="exp-1",
        target_version="candidate-version",
        seed_version="seed-version",
        attempt=attempt,
        cfg=cfg,
        store=store,
        mark_material_func=lambda node, status: marked.append((node, status)),
    )
    second = record_rejected_subject_distillation(
        node_id="computing.files",
        family="file",
        experiment_id="exp-1",
        target_version="candidate-version",
        seed_version="seed-version",
        attempt=attempt,
        cfg=cfg,
        store=store,
        mark_material_func=lambda node, status: marked.append((node, status)),
    )
    assert len(first) == 1
    assert second == []
    assert marked == [("computing.files", "verified_packet")]
    rows = store.approved_experiences(limit=10, minimum_quality=0.8)
    assert len(rows) == 1
    assert "rejected_subject_credit" in rows[0][2]


def test_rejected_subject_distillation_uses_default_store(monkeypatch, tmp_path):
    import butterfly.learning.skill_credit as skill_credit

    store = MemoryStore(tmp_path / "memory.db")
    monkeypatch.setattr(skill_credit, "MemoryStore", lambda: store)
    ids = record_rejected_subject_distillation(
        node_id="programming.api_basics",
        family="api",
        experiment_id="exp-default-store",
        target_version="candidate-version",
        seed_version="seed-version",
        attempt={"alpha": 0.75, "skill_score": 1.0, "skill_delta": 0.25},
        cfg={"rejected_subject_distillation": {"enabled": True}},
        mark_material_func=lambda *_: None,
    )
    assert len(ids) == 1
    assert store.approved_experiences(limit=10, minimum_quality=0.8)
