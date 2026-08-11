from butterfly.pipeline import _first_incomplete, STAGE_NAMES


def state(statuses):
    return {
        "stages": {
            name: {"status": statuses.get(name, "pending")}
            for name in STAGE_NAMES
        }
    }


def test_auto_resume_primitive_skips_completed_stages():
    value = state({
        "prepare": "complete",
        "build_dataset": "complete",
        "train": "pending",
    })
    assert _first_incomplete(value) == "train"


def test_all_complete_has_no_resume_stage():
    value = state({name: "complete" for name in STAGE_NAMES})
    assert _first_incomplete(value) is None
