import io

import pytest

from butterfly.learning.autonomy import (
    _Tee,
    _interrupted_block_fields,
    _is_training_stop,
    _recovery_artifacts,
    _recovery_lesson,
    _recovery_steps,
)


def test_recovery_steps_cover_every_nonterminal_deliberate_status():
    assert _recovery_steps("planned") == ("prepare", "build", "train", "evaluate")
    assert _recovery_steps("prepared") == ("build", "train", "evaluate")
    assert _recovery_steps("dataset_ready") == ("train", "evaluate")
    assert _recovery_steps("candidate_ready") == ("evaluate",)
    assert _recovery_steps("rejected") == ()
    assert _recovery_steps("lab_accepted") == ()
    assert _recovery_steps("promoted") == ()


def test_recovery_steps_reject_unknown_nonterminal_status():
    with pytest.raises(RuntimeError):
        _recovery_steps("mystery_partial_state")


def test_recovery_lesson_preserves_lifelong_identity_and_strategy():
    exp = {
        "recipe_name": "autonomy_instruction",
        "focus_target": {
            "family": "two_steps",
            "study_metric": "instruction_two_steps_component",
            "reason": "weakest_family",
            "curriculum_node": "instruction.two_steps",
            "dynamic_family": "two_steps",
            "strategy_id": "focused",
            "corpus_aliases": ["instruction:two"],
        },
    }
    lesson = _recovery_lesson(exp)
    assert lesson["capability"] == "instruction_format"
    assert lesson["curriculum_node"] == "instruction.two_steps"
    assert lesson["dynamic_family"] == "two_steps"
    assert lesson["strategy_id"] == "focused"
    assert lesson["focus_reason"] == "weakest_family"


def test_recovery_artifacts_requires_matching_manifest_and_resume_pair(tmp_path, monkeypatch):
    import butterfly.learning.autonomy as ns

    monkeypatch.setattr(ns, "ROOT", tmp_path)
    target = "target-next"
    exp = {
        "experiment_id": "abc123",
        "target_version": target,
        "recipe_hash": "recipehash",
        "suite_id": "suite-demo",
    }

    corpus = tmp_path / "data" / "corpus" / "deliberate"
    corpus.mkdir(parents=True)
    (corpus / "manifest.json").write_text(
        """{
          "experiment_id": "abc123",
          "target_version": "target-next",
          "recipe_hash": "recipehash",
          "suite_id": "suite-demo"
        }""",
        encoding="utf-8",
    )

    state = tmp_path / "training_state" / "deliberate"
    state.mkdir(parents=True)
    (state / "progress.json").write_text(
        """{
          "experiment_id": "abc123",
          "target_version": "target-next",
          "recipe_hash": "recipehash",
          "stage": "instruction_focus_v2",
          "stage_complete": false,
          "epoch": 2,
          "step": 120
        }""",
        encoding="utf-8",
    )
    (state / "resume.safetensors").write_bytes(b"not-real-weights-unit-test")

    info = _recovery_artifacts(exp)
    assert info["manifest_matches"] is True
    assert info["resume_checkpoint_matches"] is True
    assert info["resume_stage"] == "instruction_focus_v2"
    assert info["resume_epoch"] == 2
    assert info["resume_step"] == 120


class _BrokenStream:
    def write(self, _):
        raise OSError("simulated disk disconnect")

    def flush(self):
        raise OSError("simulated disk disconnect")


def test_tee_keeps_console_alive_when_log_stream_disconnects():
    healthy = io.StringIO()
    tee = _Tee(healthy, _BrokenStream())
    assert tee.write("still visible") == len("still visible")
    tee.flush()
    assert healthy.getvalue() == "still visible"


def test_training_stop_is_recorded_as_interrupted_block():
    TrainingStopRequested = type("TrainingStopRequested", (Exception,), {})
    stop = TrainingStopRequested("stop")
    stop.stage = "comprehension_repair_v2"
    stop.epoch = 2
    stop.step = 174

    assert _is_training_stop(stop)
    fields = _interrupted_block_fields(stop)
    assert fields["status"] == "interrupted"
    assert fields["stop_reason"] == "stop_file"
    assert fields["resume_stage"] == "comprehension_repair_v2"
    assert fields["resume_epoch"] == 2
    assert fields["resume_step"] == 174
