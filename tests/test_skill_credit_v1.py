from butterfly.learning.skill_credit import (
    _validated_alpha_ladder,
    dynamic_credit_gate,
    fixed_safety_check,
)


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


def test_alpha_ladder_is_largest_safe_fraction_first():
    assert _validated_alpha_ladder(
        {"alpha_ladder": [0.35, 1.0, 0.75, 0.5, 0.75]}
    ) == [1.0, 0.75, 0.5, 0.35]
