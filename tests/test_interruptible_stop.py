import inspect


def test_validation_loss_accepts_stop_callback():
    from butterfly.deliberate_trainer import _validation_loss

    assert "stop_requested" in inspect.signature(_validation_loss).parameters


def test_study_exam_checks_stop_between_cases(monkeypatch):
    from butterfly.learning import study_exam

    calls = {"count": 0}

    def fake_generate(*_args, **_kwargs):
        return "User: prompt\nButterfly: ok"

    def stop_requested():
        calls["count"] += 1
        return calls["count"] > 1

    monkeypatch.setattr(study_exam, "generate", fake_generate)

    try:
        study_exam.study_microbenchmark(object(), object(), stop_requested=stop_requested)
    except StopIteration:
        pass
    else:
        raise AssertionError("study_microbenchmark should stop between cases")


def test_dynamic_exam_checks_stop_between_cases(monkeypatch):
    from butterfly.learning import dynamic_exam

    calls = {"count": 0}
    bank = dynamic_exam.generate_bank("file", 123, count=2, mode="selection")

    def fake_generate(*_args, **_kwargs):
        return "User: prompt\nButterfly: Un archivo guarda datos con nombre."

    def stop_requested():
        calls["count"] += 1
        return calls["count"] > 1

    monkeypatch.setattr(dynamic_exam, "generate", fake_generate)

    try:
        dynamic_exam.evaluate_bank(object(), object(), bank, stop_requested=stop_requested)
    except StopIteration:
        pass
    else:
        raise AssertionError("evaluate_bank should stop between cases")
