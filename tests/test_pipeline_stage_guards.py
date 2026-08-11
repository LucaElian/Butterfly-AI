from butterfly.pipeline import STAGE_NAMES, _missing_prerequisites


def _state(**statuses):
    return {"stages": {name: {"status": statuses.get(name, "pending")} for name in STAGE_NAMES}}


def test_manual_build_requires_prepare():
    assert _missing_prerequisites(_state(), "build_dataset") == ("prepare",)


def test_manual_train_requires_dataset_after_prepare():
    assert _missing_prerequisites(_state(prepare="complete"), "train") == ("build_dataset",)


def test_manual_evaluate_requires_train_after_prior_stages():
    state = _state(prepare="complete", build_dataset="complete")
    assert _missing_prerequisites(state, "evaluate_and_promote") == ("train",)


def test_manual_evaluate_allowed_when_prior_stages_complete():
    state = _state(prepare="complete", build_dataset="complete", train="complete")
    assert _missing_prerequisites(state, "evaluate_and_promote") == ()
