import json
from butterfly.config import ROOT
from butterfly.learning.evaluator import _semantic_score
from butterfly.learning.night_study import _family_target_from_failure, _weakest_study_family
from butterfly.corpus.deliberate import _row_matches_focus
from butterfly.deliberate_trainer import _failure_driven_stage_cfg


def test_mixed_greeting_state_validator_accepts_either_response():
    case = {"validator": "greeting_or_state"}
    assert _semantic_score("Todo bien. Estoy lista para ayudarte.", case) == 1.0
    assert _semantic_score("¡Buenas! Decime qué querés hacer.", case) == 1.0
    assert _semantic_score("Un archivo guarda datos.", case) == 0.0


def test_failure_family_selected_by_validator_not_case_id():
    spec = {"family_targets": {
        "file": {"validators": ["definition_file"], "study_metric": "retention_file_component"},
        "folder": {"validators": ["definition_folder"], "study_metric": "retention_folder_component"},
    }}
    family, target = _family_target_from_failure(spec, [{"id": "anything", "validator": "definition_folder"}])
    assert family == "folder"
    assert target["study_metric"] == "retention_folder_component"


def test_weakest_instruction_family_comes_from_shadow():
    spec = {"family_targets": {
        "sentence": {"study_metric": "instruction_sentence_component"},
        "two_steps": {"study_metric": "instruction_two_steps_component"},
        "missing": {"study_metric": "instruction_missing_component"},
        "short": {"study_metric": "instruction_short_component"},
    }}
    family, target = _weakest_study_family(spec, {
        "instruction_sentence_component": 0.50,
        "instruction_two_steps_component": 0.28,
        "instruction_missing_component": 0.44,
        "instruction_short_component": 0.41,
    })
    assert family == "two_steps"
    assert target["study_metric"] == "instruction_two_steps_component"


def test_focus_alias_matching_is_generic():
    assert _row_matches_focus({"family":"concept:carpeta:train:1","skill":"definition:carpeta"}, ["folder","carpeta"])
    assert _row_matches_focus({"family":"route:folder:train:1","skill":"intent:folder"}, ["folder","carpeta"])
    assert not _row_matches_focus({"family":"route:file:train:1","skill":"intent:file"}, ["folder","carpeta"])


def test_trainer_redirects_checkpoint_focus_to_family():
    stage = {
        "study_focus_metrics": ["retention_comprehension_component"],
        "study_protected_metrics": ["retention_file_component","retention_folder_component","retention_api_component"],
        "failure_focus_min_delta": 0.02,
        "min_each_study_focus_delta": 0.0,
    }
    exp = {"focus_target": {"family":"folder","study_metric":"retention_folder_component","reason":"critical_failure"}}
    cfg = _failure_driven_stage_cfg(stage, exp)
    assert cfg["study_focus_metrics"] == ["retention_folder_component"]
    assert "retention_folder_component" not in cfg["study_protected_metrics"]
    assert "retention_file_component" in cfg["study_protected_metrics"]
    assert cfg["min_each_study_focus_delta"] == 0.02


def test_v3_config_has_family_targets():
    cfg = json.loads((ROOT / "config" / "night_study.json").read_text(encoding="utf-8"))
    # Failure-driven family targeting was introduced in V3 and remains a required contract.
    assert cfg["engine_version"] >= 3
    assert "folder" in cfg["curriculum"]["comprehension"]["family_targets"]
    assert "two_steps" in cfg["curriculum"]["instruction_format"]["family_targets"]
