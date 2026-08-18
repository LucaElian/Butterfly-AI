from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import re

from ..config import ROOT, load_json, project_relpath
from ..learning.evaluator import (
    BENCHMARK_RESERVED_EXACT_TARGETS,
    BENCHMARK_RESERVED_FICTIONAL,
    benchmark_surface_prompts,
    normalize_surface,
)
from ..learning.study_exam import study_surface_prompts
from .focus_packets import build_focus_packets
from .verified_experiences import build_verified_experience_packets
from .skills import BUILDERS
from .skills.common import contains_mojibake, dedupe, sample

DATA_ROOT = ROOT / "data" / "corpus" / "deliberate"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
HISTORY_PATH = DATA_ROOT / "history.json"
AUTONOMY_LEARNING_CONFIG_PATH = ROOT / "config" / "autonomy_learning.json"


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
    held_out |= {normalize_surface(prompt) for prompt in study_surface_prompts()}
    return [row for row in rows if normalize_surface(row["user"]) not in held_out]


def _validate_rows(stage: str, train: list[dict], valid: list[dict]):
    overlap = {x["family"] for x in train} & {x["family"] for x in valid}
    if overlap:
        raise RuntimeError(f"Train/valid family overlap in {stage}: {sorted(overlap)[:10]}")

    benchmark_prompts = benchmark_surface_prompts()
    benchmark_prompts |= {normalize_surface(prompt) for prompt in study_surface_prompts()}
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


def _focus_aliases(experiment: dict) -> list[str]:
    target = experiment.get("focus_target") or {}
    return [str(v).casefold() for v in target.get("corpus_aliases", []) if str(v).strip()]


def _verified_experience_settings() -> dict:
    cfg = load_json(AUTONOMY_LEARNING_CONFIG_PATH, {})
    value = cfg.get("verified_experiences") if isinstance(cfg, dict) else None
    return dict(value or {})


def _row_matches_focus(row: dict, aliases: list[str]) -> bool:
    if not aliases:
        return False
    haystack = (str(row.get("family", "")) + " " + str(row.get("skill", ""))).casefold()
    return any(alias in haystack for alias in aliases)


def _boost_focus_rows(rows: list[dict], aliases: list[str], repeat: int, seed: int) -> tuple[list[dict], int]:
    if not aliases or repeat <= 1:
        return list(rows), 0
    focus_rows = [row for row in rows if _row_matches_focus(row, aliases)]
    if not focus_rows:
        return list(rows), 0
    boosted = list(rows)
    for _ in range(repeat - 1):
        boosted.extend(dict(row) for row in focus_rows)
    random.Random(seed).shuffle(boosted)
    return boosted, len(focus_rows)


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
    print("Offline corpus build only: no live Internet or teacher call during this stage.")

    verified_cfg = _verified_experience_settings()
    verified_enabled = bool(verified_cfg.get("automatic_training_enabled", False))
    verified_limit = int(verified_cfg.get("max_packets_per_stage", 32))
    verified_min_quality = float(verified_cfg.get("minimum_quality", 0.7))
    used_verified_experience_ids: set[int] = set()

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

        target = experiment.get("focus_target") or {}
        packet_rows = []
        packet_family = target.get("dynamic_family")
        packet_count = int(target.get("learning_packet_cases", 0) or 0)
        if packet_family and packet_count > 0:
            packet_seed = f"{experiment['random_seed']}:{stage}:focus-packets"
            packet_rows = _filter_benchmark_surfaces(
                build_focus_packets(str(packet_family), packet_seed, count=packet_count)
            )
            train.extend(packet_rows)

        verified_rows = []
        verified_ids: list[int] = []
        if verified_enabled:
            verified_rows, verified_ids = build_verified_experience_packets(
                target,
                limit=verified_limit,
                minimum_quality=verified_min_quality,
            )
            train.extend(_filter_benchmark_surfaces(verified_rows))
            used_verified_experience_ids.update(verified_ids)

        train = dedupe(train, normalize_surface)
        valid = dedupe(valid, normalize_surface)
        _validate_rows(stage, train, valid)

        focus_aliases = _focus_aliases(experiment)
        focus_repeat = int(stage_cfg.get("focus_repeat", 1))
        focus_repeat += int((experiment.get("focus_target") or {}).get("focus_repeat_delta", 0))
        focus_repeat = max(1, focus_repeat)
        train, focus_unique_rows = _boost_focus_rows(train, focus_aliases, focus_repeat, int(experiment["random_seed"]) + stage_index * 31337)
        focus_packet_rows = sum(1 for row in train if row.get("source") == "focus_packet")
        verified_experience_rows = sum(1 for row in train if row.get("source") == "verified_experience")

        train_path, valid_path = stage_files(stage)
        _write_jsonl(train_path, train)
        _write_jsonl(valid_path, valid)

        manifest["stages"][stage] = {
            "train_rows": len(train),
            "valid_rows": len(valid),
            "train_families": len({x["family"] for x in train}),
            "valid_families": len({x["family"] for x in valid}),
            "train_skills": dict(Counter(x["skill"] for x in train)),
            "focus_family": (experiment.get("focus_target") or {}).get("family"),
            "focus_aliases": focus_aliases,
            "focus_repeat": focus_repeat,
            "focus_unique_rows": focus_unique_rows,
            "focus_packet_rows": focus_packet_rows,
            "verified_experience_rows": verified_experience_rows,
            "train_file": project_relpath(train_path),
            "valid_file": project_relpath(valid_path),
            "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "valid_sha256": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
        }
        print(
            f"{stage:24s} train {len(train):,} | valid {len(valid):,} | "
            f"families {manifest['stages'][stage]['train_families']}/"
            f"{manifest['stages'][stage]['valid_families']}"
        )

    manifest["verified_experience_ids"] = sorted(used_verified_experience_ids)
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
        "verified_experience_ids": sorted(used_verified_experience_ids),
    })
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {MANIFEST_PATH}")
    return manifest




def verified_experience_ids_for_experiment(experiment_id: str) -> list[int]:
    history = load_json(HISTORY_PATH, {"format": 2, "builds": []})
    if not isinstance(history, dict):
        return []
    matches = [
        row for row in history.get("builds") or []
        if str(row.get("experiment_id") or "") == str(experiment_id)
    ]
    ids: set[int] = set()
    for row in matches:
        for value in row.get("verified_experience_ids") or []:
            ids.add(int(value))
    return sorted(ids)

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing deliberate corpus manifest: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
