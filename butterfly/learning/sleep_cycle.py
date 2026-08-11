from __future__ import annotations

from copy import deepcopy
import json

from ..memory import MemoryStore
from ..checkpoint import load_entry, save_stable_model
from ..config import MODELS_DIR, CORPUS_DIR, ROOT, load_pipeline_config
from ..experiments import create_experiment, load_current_experiment, load_recipe, mark_experiment_status
from ..registry import get_entry, register_model
from ..training.runtime import continue_training, best_device
from ..upgrade import evaluate_candidate


def experiences_to_text(rows):
    chunks, ids = [], []
    for row in rows:
        id_, task, context, actions_json, result, lesson, quality = row
        ids.append(id_)
        actions = json.loads(actions_json or "[]")
        chunks.append(
            f"User: Tarea: {task}\nButterfly: Contexto: {context}\n"
            f"Acciones: {actions}\nResultado: {result}\nLeccion verificada: {lesson}\n<END>\n"
        )
    return "\n".join(chunks), ids


def _replay_text(limit_chars=250_000):
    chunks = []
    for name in ("conversation_train.txt", "instruction_train.txt"):
        path = CORPUS_DIR / name
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore")[: limit_chars // 2])
    return "\n".join(chunks)


def _sleep_allowed():
    return bool(load_pipeline_config().get("allow_sleep_learning", False))


def run_sleep_cycle(steps=120):
    if not _sleep_allowed():
        print("Sleep learning is paused by pipeline configuration.")
        print("Verified experiences remain stored and unused.")
        return False

    memory = MemoryStore()
    rows = memory.approved_experiences()
    if not rows:
        print("No verified high-quality unused experiences.")
        return False

    current = load_current_experiment()
    if current and current.get("status") not in {"promoted", "lab_accepted", "rejected", "cancelled"}:
        print("Sleep learning skipped: another experiment is currently in progress.")
        return False

    experiment = create_experiment("sleep_learning")
    recipe = load_recipe(experiment["recipe_name"])
    seed = get_entry(experiment["seed_version"])
    if not seed:
        raise RuntimeError("Sleep experiment seed disappeared.")

    model, _, tokenizer = load_entry(seed, device=best_device())
    candidate = deepcopy(model)
    new_text, ids = experiences_to_text(rows)
    candidate, loss = continue_training(
        candidate,
        _replay_text() + "\n" + new_text,
        tokenizer,
        steps=steps,
    )

    target = experiment["target_version"]
    path = MODELS_DIR / f"butterfly-v{target}-candidate.safetensors"
    tokenizer_path = (seed.get("metadata") or {}).get("tokenizer_path")
    save_stable_model(path, candidate, extra={
        "sleep_cycle": True,
        "experiment_id": experiment["experiment_id"],
        "source_experiences": ids,
        "train_loss": loss,
        "seed_brain": seed["version"],
    })
    register_model(path, target, active=False, status="candidate", metadata={
        "sleep_cycle": True,
        "experiment_id": experiment["experiment_id"],
        "source_experiences": ids,
        "train_loss": loss,
        "seed_brain": seed["version"],
        "tokenizer_path": tokenizer_path,
    })
    mark_experiment_status("candidate_ready")

    result, report_path, metrics = evaluate_candidate(target, recipe)
    if result in {"PROMOTED", "LAB_ACCEPTED"}:
        memory.mark_used(ids)
    mark_experiment_status(
        {"PROMOTED": "promoted", "LAB_ACCEPTED": "lab_accepted", "REJECTED": "rejected"}[result],
        evaluation_report=str(report_path.relative_to(ROOT)),
        final_score=metrics.get("score"),
    )
    return result != "REJECTED"
