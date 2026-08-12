from butterfly.learning.evaluator import CASES
from butterfly.learning.night_study import (
    _critical_cases,
    _family_target_from_failure,
    load_night_config,
)


def _critical_folder_spec():
    return next(
        case for case in CASES
        if case.get("critical")
        and case.get("category") == "comprehension"
        and case.get("validator") == "definition_folder"
    )


def test_critical_case_recovers_canonical_validator_when_result_row_is_compact():
    case = _critical_folder_spec()
    metrics = {
        "critical_failures": [case["id"]],
        "cases": [{"id": case["id"], "category": "comprehension"}],
    }
    rows = _critical_cases(metrics, "comprehension")
    assert len(rows) == 1
    assert rows[0]["validator"] == "definition_folder"


def test_critical_folder_validator_maps_to_configured_folder_focus():
    case = _critical_folder_spec()
    metrics = {
        "critical_failures": [case["id"]],
        "cases": [{"id": case["id"], "category": "comprehension"}],
    }
    rows = _critical_cases(metrics, "comprehension")
    spec = load_night_config()["curriculum"]["comprehension"]
    family, target = _family_target_from_failure(spec, rows)
    assert family == "folder"
    assert target["study_metric"] == "retention_folder_component"
    assert set(target["corpus_aliases"]) >= {"folder", "carpeta"}
