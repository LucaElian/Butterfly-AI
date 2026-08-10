from __future__ import annotations
from pathlib import Path
import json, shutil, time
from .config import ROOT, LEGACY_TOKENIZER_PATH, TOKENIZERS_DIR, BENCHMARKS_DIR, V4_TOKENIZER_PATH, ensure_dirs
from .registry import (
    get_active_entry, update_entry, get_candidate_entry, promote, compact_to_active,
    load_registry, save_registry, append_history,
)
from .checkpoint import load_entry, load_checkpoint, save_stable_model, move_model_artifacts, delete_model_artifacts, metadata_path
from .trainer import best_device
from .learning.evaluator import behavior_benchmark, print_benchmark


def _compact_legacy_active_checkpoint(entry):
    """Strip the v0.0003 Adam state without changing the neural weights.

    The original .pt is only deleted after a safetensors round-trip is verified.
    This is storage compaction, not a model upgrade.
    """
    from .config import MODELS_DIR
    old = MODELS_DIR / entry["path"]
    if old.suffix != ".pt" or not old.exists():
        return entry
    version = entry["version"]
    lean = MODELS_DIR / f"butterfly-v{version}.safetensors"
    print(f"Compacting legacy v{version} checkpoint before v0.0004 training...")
    print(f"Legacy checkpoint: {old.stat().st_size/1024/1024:.1f} MB")
    model, payload, _ = load_entry(entry, device="cpu")
    save_stable_model(lean, model, extra={
        "version": version,
        "storage_migration": "legacy-pt-to-safetensors",
        "optimizer_removed": "optimizer" in payload,
        "legacy_extra": payload.get("extra") or {},
    })
    verify_model, _ = load_checkpoint(lean, device="cpu")
    original = model.state_dict(); restored = verify_model.state_dict()
    if original.keys() != restored.keys() or any(not original[k].equal(restored[k]) for k in original):
        delete_model_artifacts(lean)
        raise RuntimeError("Legacy checkpoint compaction verification failed. Original .pt was NOT deleted.")
    update_entry(version, path=lean.name, metadata={
        "storage_format": "safetensors-weights-only",
        "optimizer_included": False,
        "compacted_from": old.name,
    })
    old.unlink(missing_ok=True)
    print(f"Verified lean checkpoint: {lean.stat().st_size/1024/1024:.1f} MB")
    print("Legacy .pt removed locally only after exact weight verification.")
    return get_active_entry()


def prepare_v0004():
    ensure_dirs()
    entry = get_active_entry()
    if not entry:
        raise RuntimeError("No hay Butterfly activa. v0.0004 espera actualizar la instalacion permanente que ya tiene v0.0003.")
    version = entry["version"]
    meta = entry.get("metadata") or {}
    if not meta.get("tokenizer_path"):
        if not LEGACY_TOKENIZER_PATH.exists():
            raise FileNotFoundError(LEGACY_TOKENIZER_PATH)
        snap = TOKENIZERS_DIR / f"tokenizer-v2-v{version}.json"
        shutil.copy2(LEGACY_TOKENIZER_PATH, snap)
        update_entry(version, metadata={"tokenizer_path": str(snap.relative_to(ROOT))})
        entry = get_active_entry()
        print(f"Old tokenizer preserved as {snap}")
    bench = BENCHMARKS_DIR / f"baseline-v{version}-v0004-suite.json"
    if bench.exists():
        print(f"Baseline already exists: {bench}")
        metrics = json.loads(bench.read_text(encoding="utf-8"))
        _compact_legacy_active_checkpoint(get_active_entry())
        return metrics
    print(f"Evaluating current active Butterfly v{version} before changing anything...")
    model, _, tok = load_entry(entry, device=best_device())
    metrics = behavior_benchmark(model, tok)
    bench.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print_benchmark(metrics)
    print(f"Baseline saved: {bench}")
    append_history(version, "baseline", score=metrics.get("score"), metadata={
        "language_component": metrics.get("language_component"),
        "benchmark": str(bench.relative_to(ROOT)),
    })
    _compact_legacy_active_checkpoint(get_active_entry())
    return metrics


def _remove_registry_version(version: str):
    reg = load_registry()
    reg["versions"] = [x for x in reg.get("versions", []) if x["version"] != version]
    save_registry(reg)


def compare_and_promote(candidate_version="0.0004"):
    active = get_active_entry()
    candidate = get_candidate_entry(candidate_version)
    if not active:
        raise RuntimeError("No active baseline model")
    if not candidate:
        raise RuntimeError("No v0.0004 candidate. Run training first.")

    baseline_path = BENCHMARKS_DIR / f"baseline-v{active['version']}-v0004-suite.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        bm, _, bt = load_entry(active, device=best_device())
        baseline = behavior_benchmark(bm, bt)

    print(f"\nEvaluating candidate v{candidate_version}...")
    cm, _, ct = load_entry(candidate, device=best_device())
    cand = behavior_benchmark(cm, ct)
    print_benchmark(cand)
    report = {
        "baseline_version": active["version"],
        "candidate_version": candidate_version,
        "baseline": baseline,
        "candidate": cand,
        "created_at": time.time(),
    }
    report_path = BENCHMARKS_DIR / f"comparison-v{active['version']}-vs-v{candidate_version}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    improvement = cand["score"] - baseline["score"]
    passes = (
        improvement >= 0.03
        and cand["language_component"] >= baseline["language_component"] + 0.03
        and cand["conversation_component"] >= baseline["conversation_component"] - 0.02
        and cand["epistemic_engine_component"] >= 0.99
    )
    print(f"\nBaseline overall : {baseline['score']:.4f}")
    print(f"Candidate overall: {cand['score']:.4f}")
    print(f"Improvement      : {improvement:+.4f}")

    from .config import MODELS_DIR, DATA_DIR
    candidate_path = MODELS_DIR / candidate["path"]

    if passes:
        canonical = MODELS_DIR / f"butterfly-v{candidate_version}.safetensors"
        if candidate_path.resolve() != canonical.resolve():
            move_model_artifacts(candidate_path, canonical)
            update_entry(candidate_version, path=canonical.name)
        update_entry(candidate_version, score=cand["score"], metadata={
            "benchmark": str(report_path.relative_to(ROOT)),
            "storage_format": "safetensors-weights-only",
            "optimizer_included": False,
        })
        promote(candidate_version)
        append_history(candidate_version, "promoted", score=cand["score"], metadata={
            "parameters": (candidate.get("metadata") or {}).get("parameters"),
            "tokenizer_vocab": (candidate.get("metadata") or {}).get("tokenizer_vocab"),
            "benchmark": str(report_path.relative_to(ROOT)),
            "brain_format": "safetensors-weights-only",
        })
        compact_to_active()
        LEGACY_TOKENIZER_PATH.unlink(missing_ok=True)
        for duplicate in (DATA_DIR / "consolidated.txt", DATA_DIR / "distilled.txt", DATA_DIR / "distilled.jsonl"):
            duplicate.unlink(missing_ok=True)
        print(f"\nPROMOTED Butterfly v{candidate_version}.")
        print("The accepted brain is inference-only safetensors: optimizer state is NOT stored in the stable model.")
        print("Old physical checkpoint/tokenizer and obsolete duplicate training files were burned AFTER promotion.")
        print("Corpus, source manifest, memory, verified knowledge, lessons, benchmarks and compact model history remain.")
        return True

    print("\nREJECTED: candidate did not beat the active Butterfly by the required margin.")
    append_history(candidate_version, "rejected", score=cand.get("score"), metadata={"benchmark": str(report_path.relative_to(ROOT))})
    delete_model_artifacts(candidate_path)
    _remove_registry_version(candidate_version)
    print("Candidate brain deleted; corpus and tokenizer kept for another attempt.")
    return False
