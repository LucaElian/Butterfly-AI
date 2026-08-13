import inspect

import pytest


class _FakeParam:
    requires_grad = True
    device = "cpu"

    def numel(self):
        return 1


class _FakeTensor:
    def to(self, _device):
        return self


class _FakeLoss:
    def backward(self):
        pass

    def item(self):
        return 0.25


class _FakeModel:
    blocks = []
    cfg = type("Cfg", (), {"vocab_size": 1})()

    def __init__(self):
        self.param = _FakeParam()

    def parameters(self):
        return iter([self.param])

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, *_args):
        return None, _FakeLoss()


class _FakeDataset:
    answer_tokens = 1

    def __init__(self, *_args, **_kwargs):
        pass


class _FakeLoader:
    def __init__(self, *_args, **_kwargs):
        pass

    def __iter__(self):
        return iter([(_FakeTensor(), _FakeTensor())])

    def __len__(self):
        return 1


class _FakeOpt:
    def zero_grad(self, **_kwargs):
        pass

    def step(self):
        pass


def _patch_light_train_stage(monkeypatch, dt):
    monkeypatch.setattr(dt, "load_jsonl", lambda _path: [{"text": "x"}])
    monkeypatch.setattr(dt, "AssistantOnlyDialogueDataset", _FakeDataset)
    monkeypatch.setattr(dt, "_set_frozen_blocks", lambda *_args: (1, 1, 0))
    monkeypatch.setattr(dt.torch.optim, "AdamW", lambda *_args, **_kwargs: _FakeOpt())
    monkeypatch.setattr(dt.torch.nn.utils, "clip_grad_norm_", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dt, "DataLoader", _FakeLoader)
    monkeypatch.setattr(dt, "_print_study", lambda *_args, **_kwargs: None)


def _stage_inputs():
    experiment = {
        "experiment_id": "exp-test",
        "target_version": "0.1-test",
        "recipe_hash": "recipe-test",
        "random_seed": 7,
    }
    stage_cfg = {
        "name": "focus_stage",
        "seq_len": 8,
        "batch_size": 1,
        "max_epochs": 1,
        "lr": 0.001,
        "study_focus_metrics": ["study_score"],
        "study_protected_metrics": [],
    }
    manifest_stage = {"train_file": "train.jsonl", "valid_file": "valid.jsonl"}
    return experiment, stage_cfg, manifest_stage


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


def test_entry_exam_stop_records_replayable_resume_phase(monkeypatch):
    from butterfly import deliberate_trainer as dt

    _patch_light_train_stage(monkeypatch, dt)
    saved = {}

    def stop_in_entry(*_args, **_kwargs):
        raise StopIteration("stop entry")

    def save_resume(_model, progress, _message):
        saved["progress"] = progress

    monkeypatch.setattr(dt, "_study_with_dynamic_focus", stop_in_entry)
    monkeypatch.setattr(dt, "_save_resume", save_resume)

    experiment, stage_cfg, manifest_stage = _stage_inputs()

    with pytest.raises(dt.TrainingStopRequested):
        dt._train_stage(
            _FakeModel(), object(), experiment, stage_cfg, manifest_stage,
            resume_progress=None, completed=[], stop_requested=lambda: False,
        )

    assert saved["progress"]["resume_phase"] == dt.RESUME_PHASE_ENTRY_EXAM
    assert saved["progress"]["epoch"] == 1
    assert saved["progress"]["step"] == 0
    assert "stage_entry_study" not in saved["progress"]


def test_post_epoch_study_exam_stop_rolls_back_before_resume_save(monkeypatch):
    from butterfly import deliberate_trainer as dt

    _patch_light_train_stage(monkeypatch, dt)
    calls = []
    saved = {}
    entry_exam = {"study_score": 0.2}
    study_calls = {"count": 0}

    def study_exam(*_args, **_kwargs):
        study_calls["count"] += 1
        if study_calls["count"] == 1:
            return entry_exam
        raise StopIteration("stop post-epoch exam")

    def save_resume(_model, progress, _message):
        calls.append("save")
        saved["progress"] = progress

    monkeypatch.setattr(dt, "_study_with_dynamic_focus", study_exam)
    monkeypatch.setattr(dt, "_validation_loss", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dt, "_save_best", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dt, "_save_resume", save_resume)
    monkeypatch.setattr(dt, "_rollback_to_best_for_stop", lambda *_args, **_kwargs: calls.append("rollback"))

    experiment, stage_cfg, manifest_stage = _stage_inputs()

    with pytest.raises(dt.TrainingStopRequested):
        dt._train_stage(
            _FakeModel(), object(), experiment, stage_cfg, manifest_stage,
            resume_progress=None, completed=[], stop_requested=lambda: False,
        )

    assert calls == ["rollback", "save"]
    assert saved["progress"]["resume_phase"] == dt.RESUME_PHASE_ACCEPTED_CHECKPOINT
    assert saved["progress"]["epoch"] == 2
    assert saved["progress"]["step"] == 0
    assert saved["progress"]["best_study_exam"] == entry_exam

def test_transfer_gate_failure_keeps_selected_checkpoint_for_salvage(monkeypatch):
    from butterfly import deliberate_trainer as dt

    _patch_light_train_stage(monkeypatch, dt)
    monkeypatch.setattr(dt, "_clean_best", lambda: None)
    monkeypatch.setattr(dt, "_clean_entry", lambda: None)
    monkeypatch.setattr(dt, "_save_best", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dt, "_save_resume", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dt, "_atomic_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dt, "_validation_loss", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(dt, "_dynamic_stage_banks", lambda *_args, **_kwargs: ({"fingerprint": "selection"}, {"fingerprint": "transfer"}))
    monkeypatch.setattr(dt, "evaluate_bank", lambda *_args, **_kwargs: {"score": 0.60, "semantic": 0.0})

    calls = {"study": 0}

    def study_exam(*_args, **_kwargs):
        calls["study"] += 1
        return {
            "study_score": 0.5,
            dt.DYNAMIC_FOCUS_KEY: 0.60,
            "dynamic_selection": {"family": "file", "score": 0.60, "fingerprint": "selection"},
            "dynamic_transfer_entry": {"family": "file", "score": 0.60, "fingerprint": "transfer-entry"},
        }

    monkeypatch.setattr(dt, "_study_with_dynamic_focus", study_exam)

    experiment, stage_cfg, manifest_stage = _stage_inputs()
    experiment["focus_target"] = {
        "dynamic_family": "file",
        "checkpoint_selection_min_delta": 0.0,
        "transfer_min_delta": 0.015,
        "selection_cases": 2,
    }

    result, _completed = dt._train_stage(
        _FakeModel(), object(), experiment, stage_cfg, manifest_stage,
        resume_progress=None, completed=[], stop_requested=lambda: False,
    )

    assert result["selected_study_epoch"] == 1
    assert result["stage_rolled_back"] is False
    assert result["dynamic_transfer"]["passed"] is False
    assert calls["study"] == 2
