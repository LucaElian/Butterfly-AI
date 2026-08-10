from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from ..memory import MemoryStore
from ..checkpoint import load_active, save_stable_model, delete_model_artifacts, move_model_artifacts
from ..config import MODELS_DIR, CORPUS_DIR, ROOT
from ..registry import (
    register_model, promote, get_active_entry, compact_to_active,
    update_entry, append_history, load_registry, save_registry,
)
from ..trainer import continue_training, best_device
from .evaluator import behavior_benchmark

def experiences_to_text(rows):
    chunks = []
    ids = []
    for row in rows:
        id_, task, context, actions_json, result, lesson, quality = row
        ids.append(id_)
        actions = json.loads(actions_json or "[]")
        chunks.append(
            f"User: Tarea: {task}\nButterfly: Contexto: {context}\n"
            f"Acciones: {actions}\nResultado: {result}\nLeccion verificada: {lesson}\n<END>\n"
        )
    return "\n".join(chunks), ids

def next_version(active):
    if not active:
        return "0.0004"
    parts = active.split(".")
    width = max(4, len(parts[-1]))
    parts[-1] = f"{int(parts[-1]) + 1:0{width}d}"
    return ".".join(parts)

def _replay_text(limit_chars=250_000):
    chunks = []
    for name in ("conversation_train.txt", "instruction_train.txt"):
        p = CORPUS_DIR / name
        if p.exists():
            chunks.append(p.read_text(encoding="utf-8", errors="ignore")[: limit_chars // 2])
    return "\n".join(chunks)

def _remove_registry_version(version: str):
    reg = load_registry()
    reg["versions"] = [x for x in reg.get("versions", []) if x["version"] != version]
    save_registry(reg)

def _sleep_allowed():
    path = ROOT / "config" / "pipeline.json"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(cfg.get("allow_sleep_learning", False))

def run_sleep_cycle(steps=120):
    if not _sleep_allowed():
        print("Sleep learning is paused by the permanent pipeline configuration.")
        print("Verified experiences remain stored and unused; nothing was deleted or marked as learned.")
        return False

    memory = MemoryStore()
    rows = memory.approved_experiences()
    if not rows:
        print("No verified high-quality unused experiences. Nothing to learn tonight.")
        return False

    model, payload, tokenizer = load_active(device=best_device())
    active_entry = get_active_entry()
    active = active_entry["version"]
    baseline = behavior_benchmark(model, tokenizer)
    candidate = deepcopy(model)
    new_text, ids = experiences_to_text(rows)
    training_text = _replay_text() + "\n" + new_text
    candidate, loss = continue_training(candidate, training_text, tokenizer, steps=steps)
    metrics = behavior_benchmark(candidate, tokenizer)
    version = next_version(active)
    path = MODELS_DIR / f"butterfly-v{version}-candidate.safetensors"
    tp = (active_entry.get("metadata") or {}).get("tokenizer_path")
    save_stable_model(path, candidate, extra={
        "sleep_cycle": True,
        "source_experiences": ids,
        "baseline": baseline,
        "candidate": metrics,
        "train_loss": loss,
    })
    no_major_regression = (
        metrics["conversation_component"] >= baseline.get("conversation_component", 0.0) - 0.02
        and metrics["comprehension_component"] >= baseline.get("comprehension_component", 0.0) - 0.02
        and metrics["instruction_component"] >= baseline.get("instruction_component", 0.0) - 0.02
        and metrics["epistemic_dialogue_component"] >= baseline.get("epistemic_dialogue_component", 0.0) - 0.02
    )
    improved = (
        metrics["score"] >= baseline["score"] + 0.03
        and bool(metrics.get("promotion_eligible"))
        and no_major_regression
    )
    register_model(path, version, score=metrics["score"], active=False, metadata={
        "baseline": baseline,
        "candidate": metrics,
        "promoted": improved,
        "tokenizer_path": tp,
        "storage_format": "safetensors-weights-only",
        "optimizer_included": False,
    })
    print("Baseline:", baseline["score"], "Candidate:", metrics["score"])
    if improved:
        canonical = MODELS_DIR / f"butterfly-v{version}.safetensors"
        move_model_artifacts(path, canonical)
        update_entry(version, path=canonical.name)
        promote(version)
        append_history(version, "promoted", score=metrics["score"], metadata={
            "source": "sleep_cycle",
            "source_experiences": ids,
            "brain_format": "safetensors-weights-only",
        })
        compact_to_active()
        memory.mark_used(ids)
        print(f"PROMOTED Butterfly v{version}; old physical brain burned after evaluation.")
        return True

    append_history(version, "rejected", score=metrics["score"], metadata={"source": "sleep_cycle"})
    delete_model_artifacts(path)
    _remove_registry_version(version)
    print(f"REJECTED Butterfly v{version}; candidate deleted, memories retained.")
    return False
