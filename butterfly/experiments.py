from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import hashlib
import json
import re
import secrets
import time
import uuid

from .config import ROOT, RECIPES_DIR, load_pipeline_config, load_json, save_json
from .registry import get_seed_entry, load_history, load_registry

CURRENT_EXPERIMENT_PATH = ROOT / ".butterfly" / "current_experiment.json"
EXPERIMENT_HISTORY_PATH = ROOT / ".butterfly" / "experiment_history.json"
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")


def recipe_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid recipe name: {name!r}")
    return RECIPES_DIR / f"{name}.json"


def load_recipe(name: str) -> dict:
    path = recipe_path(name)
    value = load_json(path)
    if not isinstance(value, dict):
        raise FileNotFoundError(path)
    if value.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported recipe schema in {path}")
    if value.get("name") != name:
        raise RuntimeError(f"Recipe name mismatch in {path}")
    return value


def recipe_hash(recipe: dict) -> str:
    raw = json.dumps(recipe, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def next_version_from_values(values: list[str]) -> str:
    parsed = []
    widths = []
    for raw in values:
        text = str(raw).strip()
        match = _VERSION_RE.fullmatch(text)
        if not match:
            continue
        parsed.append(Decimal(text))
        widths.append(len(match.group(2)))
    if not parsed:
        raise RuntimeError("Cannot allocate a brain version without existing version history.")
    width = max(widths)
    step = Decimal(1).scaleb(-width)
    value = max(parsed) + step
    return f"{value:.{width}f}"


def next_brain_version() -> str:
    reg = load_registry()
    hist = load_history()
    values = [x.get("version") for x in reg.get("versions", [])]
    values += [x.get("version") for x in hist.get("versions", [])]
    return next_version_from_values([x for x in values if x])


def load_current_experiment() -> dict | None:
    value = load_json(CURRENT_EXPERIMENT_PATH)
    return value if isinstance(value, dict) else None


def save_current_experiment(value: dict):
    save_json(CURRENT_EXPERIMENT_PATH, value)


def _default_recipe_name() -> str:
    cfg = load_pipeline_config()
    name = cfg.get("default_recipe")
    if not name:
        raise RuntimeError("config/pipeline.json has no default_recipe.")
    return str(name)


def create_experiment(recipe_name: str | None = None, force: bool = False, focus_target: dict | None = None) -> dict:
    current = load_current_experiment()
    if current and current.get("status") not in {"promoted", "lab_accepted", "rejected", "cancelled"}:
        if not force:
            return current
        raise RuntimeError("A non-terminal experiment already exists.")

    seed = get_seed_entry()
    if not seed:
        raise RuntimeError("No ACTIVE or LAB seed brain is registered.")

    from .learning.evaluator import BENCHMARK_SUITE_ID

    name = recipe_name or _default_recipe_name()
    recipe = load_recipe(name)
    exp = {
        "schema_version": 1,
        "experiment_id": uuid.uuid4().hex[:12],
        "target_version": next_brain_version(),
        "seed_version": seed["version"],
        "seed_slot": "lab" if seed.get("status") == "lab" else "active",
        "recipe_name": name,
        "recipe_hash": recipe_hash(recipe),
        "suite_id": BENCHMARK_SUITE_ID,
        "random_seed": secrets.randbelow(2_000_000_000) + 1,
        "status": "planned",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    if focus_target:
        exp["focus_target"] = dict(focus_target)
    save_current_experiment(exp)
    return exp


def ensure_experiment(create: bool = True) -> dict:
    current = load_current_experiment()
    if current:
        return current
    if not create:
        raise RuntimeError("No current experiment.")
    return create_experiment()


def mark_experiment_status(status: str, **metadata):
    exp = ensure_experiment(create=False)
    exp["status"] = status
    exp["updated_at"] = time.time()
    exp.update(metadata)
    save_current_experiment(exp)

    history = load_json(EXPERIMENT_HISTORY_PATH, {"format": 1, "experiments": []})
    rows = history.setdefault("experiments", [])
    rows = [x for x in rows if x.get("experiment_id") != exp.get("experiment_id")]
    rows.append(dict(exp))
    history["experiments"] = rows
    save_json(EXPERIMENT_HISTORY_PATH, history)
    return exp


def clear_terminal_experiment():
    exp = load_current_experiment()
    if not exp:
        return
    if exp.get("status") not in {"promoted", "lab_accepted", "rejected", "cancelled"}:
        raise RuntimeError("Current experiment is not terminal.")
    CURRENT_EXPERIMENT_PATH.unlink(missing_ok=True)
