from butterfly.deliberate_trainer import (
    _study_focus_ok,
    _study_is_better,
    _study_protected_ok,
    is_better_study_checkpoint,
)

STAGE = {
    "study_focus_metrics": ["binding_exact_component", "instruction_format_component"],
    "study_protected_metrics": ["retention_conversation_component", "retention_comprehension_component"],
    "max_study_protected_regression": 0.08,
    "min_each_study_focus_delta": 0.03,
    "min_study_delta": 0.01,
}

def exam(b, i, c=.9, d=.9):
    return {
        "binding_exact_component": b,
        "instruction_format_component": i,
        "retention_conversation_component": c,
        "retention_comprehension_component": d,
        "study_score": 0.0,
    }

def test_average_focus_cannot_hide_zero_binding():
    ok, blockers = _study_focus_ok(exam(0, .8), exam(0, .2), STAGE)
    assert not ok
    assert any("binding_exact_component" in blocker for blocker in blockers)

def test_both_focus_metrics_can_pass_together():
    assert _study_focus_ok(exam(.08, .25), exam(0, .2), STAGE)[0]

def test_protected_metrics_are_checked_independently():
    ok, blockers = _study_protected_ok(exam(.1, .3, .70, .95), exam(0, .2, .95, .90), STAGE)
    assert not ok
    assert any("retention_conversation_component" in blocker for blocker in blockers)

def test_valid_loss_breaks_global_study_tie():
    entry = exam(0, .2)
    best = exam(.08, .30)
    candidate = exam(.16, .31)
    assert _study_is_better(candidate, best, entry, .7, .8, STAGE)


def test_study_checkpoint_ignores_better_dynamic_score():
    best = {
        "study": 0.4845,
        "valid_loss": 1.1085,
        "dynamic": 0.4317,
    }
    candidate = {
        "study": 0.4705,
        "valid_loss": 1.0158,
        "dynamic": 0.4661,
    }

    assert not is_better_study_checkpoint(
        candidate["study"],
        candidate["valid_loss"],
        best["study"],
        best["valid_loss"],
    )
