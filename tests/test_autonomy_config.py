import json
from pathlib import Path

from butterfly.config import ROOT
from butterfly.experiments import load_recipe
from butterfly.runtime import load_capabilities


def test_autonomy_curriculum_excludes_deterministic_capabilities():
    cfg = json.loads((ROOT / "config" / "autonomy_learning.json").read_text(encoding="utf-8"))
    curriculum = cfg["curriculum"]
    assert "exact_binding" not in curriculum
    assert "arithmetic" not in curriculum

    caps = load_capabilities()["capabilities"]
    assert caps["exact_binding"]["trainable"] is False
    assert caps["arithmetic"]["trainable"] is False


def test_autonomy_recipes_exist_and_have_no_binding_training_skill():
    for name in ("autonomy_epistemic", "autonomy_instruction"):
        recipe = load_recipe(name)
        assert recipe["training_stages"]
        skills = {
            item["name"]
            for stage in recipe["training_stages"]
            for item in stage["skills"]
        }
        assert "binding" not in skills


def test_verified_experience_autotraining_is_off_in_v1():
    cfg = json.loads((ROOT / "config" / "autonomy_learning.json").read_text(encoding="utf-8"))
    assert cfg["verified_experiences"]["automatic_training_enabled"] is False
