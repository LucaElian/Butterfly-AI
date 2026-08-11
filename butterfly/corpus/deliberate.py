from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import re

from ..config import ROOT
from ..learning.evaluator import (
    BENCHMARK_RESERVED_EXACT_TARGETS,
    BENCHMARK_RESERVED_FICTIONAL,
    benchmark_surface_prompts,
    normalize_surface,
)
from .skills import BUILDERS
from .skills.common import contains_mojibake, dedupe, sample

DATA_ROOT = ROOT / "data" / "corpus" / "deliberate"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
HISTORY_PATH = DATA_ROOT / "history.json"


def stage_files(stage_name: str):
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", stage_name).strip("-")
    if not safe:
        raise ValueError("Empty stage name")
    return DATA_ROOT / f"{safe}_train.jsonl", DATA_ROOT / f"{safe}_valid.jsonl"


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _filter_benchmark_surfaces(rows: list[dict]) -> list[dict]:
    held_out = benchmark_surface_prompts()
    return [row for row in rows if normalize_surface(row["user"]) not in held_out]


def _validate_rows(stage: str, train: list[dict], valid: list[dict]):
    overlap = {x["family"] for x in train} & {x["family"] for x in valid}
    if overlap:
        raise RuntimeError(f"Train/valid family overlap in {stage}: {sorted(overlap)[:10]}")

    benchmark_prompts = benchmark_surface_prompts()
    benchmark_values = {x.casefold() for x in BENCHMARK_RESERVED_EXACT_TARGETS}
    for item in train + valid:
        if contains_mojibake(item.get("user", "")) or contains_mojibake(item.get("assistant", "")):
            raise RuntimeError(
                f"Mojibake detected in generated corpus row for {stage}: "
                f"user={item.get('user')!r} assistant={item.get('assistant')!r}"
            )
        if normalize_surface(item["user"]) in benchmark_prompts:
            raise RuntimeError(f"Benchmark prompt leaked into {stage}: {item['user']}")
        if item.get("skill") == "binding" and item["assistant"].casefold() in benchmark_values:
            raise RuntimeError(f"Reserved exact answer leaked into {stage}: {item['assistant']}")
        low = item["user"].casefold()
        for fictional in BENCHMARK_RESERVED_FICTIONAL:
            if re.search(rf"\b{re.escape(fictional)}\b", low):
                raise RuntimeError(f"Reserved fictional entity leaked into {stage}: {fictional}")


def _clean_working_rows():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for path in DATA_ROOT.glob("*.jsonl"):
        path.unlink(missing_ok=True)


def build_corpus(experiment: dict, recipe: dict) -> dict:
    from ..learning.evaluator import BENCHMARK_SUITE_ID

    expected_hash = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if experiment.get("recipe_hash") != expected_hash:
        raise RuntimeError("Experiment recipe hash does not match current recipe.")
    if experiment.get("suite_id") != BENCHMARK_SUITE_ID:
        raise RuntimeError("Evaluator suite changed after experiment creation. Start/reprepare the experiment.")

    _clean_working_rows()
    manifest = {
        "format": 2,
        "experiment_id": experiment["experiment_id"],
        "target_version": experiment["target_version"],
        "seed_version": experiment["seed_version"],
        "recipe_name": experiment["recipe_name"],
        "recipe_hash": experiment["recipe_hash"],
        "suite_id": BENCHMARK_SUITE_ID,
        "random_seed": experiment["random_seed"],
        "benchmark_surface_leaks": 0,
        "train_valid_surface_family_overlap": 0,
        "stages": {},
    }

    print("ButterflyAI deliberate corpus")
    print(f"Experiment       : {experiment['experiment_id']}")
    print(f"Target           : {experiment['target_version']}")
    print(f"Seed             : {experiment['seed_version']}")
    print(f"Recipe           : {experiment['recipe_name']}")
    print(f"Benchmark suite  : {BENCHMARK_SUITE_ID}")
    print("Offline generation only: no Internet and no teacher model.")

    for stage_index, stage_cfg in enumerate(recipe.get("training_stages", [])):
        stage = stage_cfg["name"]
        train, valid = [], []
        for skill_index, skill_cfg in enumerate(stage_cfg.get("skills", [])):
            name = skill_cfg["name"]
            builder = BUILDERS.get(name)
            if not builder:
                raise RuntimeError(f"Unknown corpus skill: {name}")
            skill_seed = int(experiment["random_seed"]) + stage_index * 10007 + skill_index * 997
            raw_train, raw_valid = builder(skill_seed)
            raw_train = _filter_benchmark_surfaces(raw_train)
            raw_valid = _filter_benchmark_surfaces(raw_valid)
            train.extend(sample(raw_train, skill_cfg.get("train_limit"), skill_seed + 1))
            valid.extend(sample(raw_valid, skill_cfg.get("valid_limit"), skill_seed + 2))

        train = dedupe(train, normalize_surface)
        valid = dedupe(valid, normalize_surface)
        _validate_rows(stage, train, valid)

        train_path, valid_path = stage_files(stage)
        _write_jsonl(train_path, train)
        _write_jsonl(valid_path, valid)

        manifest["stages"][stage] = {
            "train_rows": len(train),
            "valid_rows": len(valid),
            "train_families": len({x["family"] for x in train}),
            "valid_families": len({x["family"] for x in valid}),
            "train_skills": dict(Counter(x["skill"] for x in train)),
            "train_file": str(train_path.relative_to(ROOT)),
            "valid_file": str(valid_path.relative_to(ROOT)),
            "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "valid_sha256": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
        }
        print(
            f"{stage:24s} train {len(train):,} | valid {len(valid):,} | "
            f"families {manifest['stages'][stage]['train_families']}/"
            f"{manifest['stages'][stage]['valid_families']}"
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    history = {"format": 2, "builds": []}
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    history["builds"] = [
        x for x in history.get("builds", [])
        if x.get("experiment_id") != experiment["experiment_id"]
    ]
    history["builds"].append({
        "experiment_id": experiment["experiment_id"],
        "target_version": experiment["target_version"],
        "suite_id": BENCHMARK_SUITE_ID,
        "recipe_hash": experiment["recipe_hash"],
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "stage_rows": {
            name: {"train": row["train_rows"], "valid": row["valid_rows"]}
            for name, row in manifest["stages"].items()
        },
    })
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {MANIFEST_PATH}")
    return manifest


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing deliberate corpus manifest: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
