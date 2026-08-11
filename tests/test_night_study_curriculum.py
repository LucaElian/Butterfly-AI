from butterfly.learning.night_study import choose_lesson


def test_curriculum_chooses_highest_priority_trainable_deficit():
    snapshot = {
        "capabilities": [
            {
                "capability": "epistemic_dialogue",
                "gap": 0.18,
                "priority": 0.25,
                "trainable": True,
                "recipe": "night_epistemic",
            },
            {
                "capability": "instruction_format",
                "gap": 0.03,
                "priority": 0.03,
                "trainable": True,
                "recipe": "night_instruction",
            },
        ]
    }
    lesson = choose_lesson(snapshot)
    assert lesson["capability"] == "epistemic_dialogue"


def test_curriculum_skips_attempted_capability():
    snapshot = {
        "capabilities": [
            {
                "capability": "epistemic_dialogue",
                "gap": 0.18,
                "priority": 0.25,
                "trainable": True,
                "recipe": "night_epistemic",
            },
            {
                "capability": "instruction_format",
                "gap": 0.03,
                "priority": 0.03,
                "trainable": True,
                "recipe": "night_instruction",
            },
        ]
    }
    lesson = choose_lesson(snapshot, {"epistemic_dialogue"})
    assert lesson["capability"] == "instruction_format"


def test_curriculum_does_not_study_satisfied_capabilities():
    snapshot = {
        "capabilities": [
            {
                "capability": "instruction_format",
                "gap": 0.0,
                "priority": 0.0,
                "trainable": True,
                "recipe": "night_instruction",
            }
        ]
    }
    assert choose_lesson(snapshot) is None
