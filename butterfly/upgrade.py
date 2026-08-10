from __future__ import annotations

import json
import time

from .config import ROOT, LEGACY_TOKENIZER_PATH, BENCHMARKS_DIR, ensure_dirs
from .registry import (
    get_active_entry,
    update_entry,
    promote,
    compact_to_active,
    load_registry,
    save_registry,
    append_history,
)
from .checkpoint import (
    load_entry,
    move_model_artifacts,
    delete_model_artifacts,
)
from .trainer import best_device
from .learning.evaluator import (
    BENCHMARK_SUITE_VERSION,
    behavior_benchmark,
    print_benchmark,
    save_benchmark,
)

def _remove_registry_version(version: str):
    reg = load_registry()
    reg["versions"] = [x for x in reg.get("versions", []) if x["version"] != version]
    save_registry(reg)

def _resolve_candidate(candidate_version: str | None):
    reg = load_registry()
    active_version = reg.get("active")
    rows = reg.get("versions", [])
    if candidate_version:
        candidate = next((x for x in rows if x.get("version") == candidate_version), None)
        if not candidate:
            raise RuntimeError(f"Candidate v{candidate_version} is not registered.")
        if candidate.get("version") == active_version or candidate.get("status") == "active":
            raise RuntimeError(
                f"Safety stop: v{candidate_version} is already the ACTIVE brain. "
                "An active brain can never be treated as a rejectable candidate."
            )
        if candidate.get("status") != "candidate":
            raise RuntimeError(
                f"v{candidate_version} has status {candidate.get('status')!r}, not 'candidate'."
            )
        return candidate

    candidates = [
        x for x in rows
        if x.get("status") == "candidate" and x.get("version") != active_version
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        versions = ", ".join(x.get("version", "?") for x in candidates)
        raise RuntimeError(
            f"More than one candidate is registered ({versions}). "
            "Specify the candidate version explicitly."
        )
    return candidates[0]

def _strict_baseline(active):
    path = BENCHMARKS_DIR / (
        f"baseline-v{active['version']}-suite-v{BENCHMARK_SUITE_VERSION}.json"
    )
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("suite_version") == BENCHMARK_SUITE_VERSION:
            return value, path

    print(
        f"No strict v{BENCHMARK_SUITE_VERSION} baseline for active v{active['version']}; "
        "evaluating it now..."
    )
    model, _, tok = load_entry(active, device=best_device())
    metrics = behavior_benchmark(model, tok)
    save_benchmark(path, metrics)
    return metrics, path

def prepare_target(target_version: str, expected_active: str | None = None):
    """Generic read-only preflight for any future deliberate candidate."""
    ensure_dirs()
    active = get_active_entry()
    if not active:
        raise RuntimeError("No active Butterfly brain.")

    if target_version == active["version"]:
        raise RuntimeError(
            f"Target v{target_version} is already active; a new candidate must use a distinct version."
        )
    if expected_active is not None and active["version"] != expected_active:
        raise RuntimeError(
            f"Expected active seed v{expected_active}, but active is v{active['version']}. "
            "Nothing was changed."
        )

    candidates = [
        x for x in load_registry().get("versions", [])
        if x.get("status") == "candidate" and x.get("version") != active["version"]
    ]
    if candidates:
        names = ", ".join("v" + str(x.get("version")) for x in candidates)
        raise RuntimeError(
            f"A candidate is already registered ({names}). Finish/reject it before stacking experiments."
        )

    model, _, tokenizer = load_entry(active, device="cpu")
    tokenizer_rel = (active.get("metadata") or {}).get("tokenizer_path")
    if not tokenizer_rel:
        raise RuntimeError("Active brain has no tokenizer lineage in registry metadata.")
    tokenizer_path = ROOT / tokenizer_rel
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    if model.cfg.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("Active model/tokenizer vocabulary mismatch. Nothing was changed.")

    baseline, baseline_path = _strict_baseline(active)
    result = {
        "target_version": target_version,
        "active_seed": active["version"],
        "parameters": model.parameter_count(),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab": tokenizer.vocab_size,
        "benchmark_suite": BENCHMARK_SUITE_VERSION,
        "baseline_score": baseline.get("score"),
        "critical_pass_rate": baseline.get("critical_pass_rate"),
        "baseline_path": str(baseline_path),
    }
    print("ButterflyAI generic target preflight OK.")
    print(f"Target brain           : v{target_version}")
    print(f"Active seed brain      : v{active['version']} (READ ONLY until promotion)")
    print(f"Parameters             : {model.parameter_count():,}")
    print(f"Tokenizer              : {tokenizer_path}")
    print(f"Tokenizer vocab        : {tokenizer.vocab_size:,}")
    print(f"Strict benchmark suite : v{BENCHMARK_SUITE_VERSION}")
    print(f"Strict baseline score  : {float(baseline.get('score', 0.0)):.4f}")
    print(f"Critical pass rate     : {float(baseline.get('critical_pass_rate', 0.0)):.4f}")
    print(f"Baseline file          : {baseline_path}")
    print("No weights, tokenizer, memory, corpus or benchmark were modified.")
    return result

def compare_and_promote(candidate_version: str | None = None):
    """Compare a distinct candidate against the active brain using strict hard gates."""
    active = get_active_entry()
    if not active:
        raise RuntimeError("No active baseline model.")

    candidate = _resolve_candidate(candidate_version)
    if candidate is None:
        print("No candidate brain is registered. Nothing to compare or delete.")
        print(f"Active brain remains Butterfly v{active['version']}.")
        return False

    candidate_version = candidate["version"]
    meta = candidate.get("metadata") or {}
    seed_brain = meta.get("seed_brain")
    if seed_brain is not None and seed_brain != active["version"]:
        raise RuntimeError(
            f"Safety stop: v{candidate_version} lineage says seed={seed_brain!r}, "
            f"but active baseline is v{active['version']}. Nothing was promoted or deleted."
        )

    candidate_tokenizer = meta.get("tokenizer_path")
    active_tokenizer = (active.get("metadata") or {}).get("tokenizer_path")
    if candidate_tokenizer is not None and candidate_tokenizer != active_tokenizer:
        raise RuntimeError(
            f"Safety stop: v{candidate_version} must inherit the exact accepted tokenizer. "
            "Tokenizer lineage differs; nothing was promoted or deleted."
        )
    if candidate_version == active["version"]:
        raise RuntimeError("Safety stop: candidate and active versions are identical.")

    baseline, _ = _strict_baseline(active)

    print(f"\nEvaluating candidate v{candidate_version} with suite v{BENCHMARK_SUITE_VERSION}...")
    cm, _, ct = load_entry(candidate, device=best_device())
    cand = behavior_benchmark(cm, ct)
    print_benchmark(cand)

    report = {
        "suite_version": BENCHMARK_SUITE_VERSION,
        "baseline_version": active["version"],
        "candidate_version": candidate_version,
        "baseline": baseline,
        "candidate": cand,
        "created_at": time.time(),
    }
    report_path = BENCHMARKS_DIR / (
        f"comparison-v{active['version']}-vs-v{candidate_version}"
        f"-suite-v{BENCHMARK_SUITE_VERSION}.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    improvement = cand["score"] - baseline["score"]
    no_major_regression = (
        cand["conversation_component"] >= baseline.get("conversation_component", 0.0) - 0.02
        and cand["comprehension_component"] >= baseline.get("comprehension_component", 0.0) - 0.02
        and cand["instruction_component"] >= baseline.get("instruction_component", 0.0) - 0.02
        and cand["epistemic_dialogue_component"]
        >= baseline.get("epistemic_dialogue_component", 0.0) - 0.02
    )
    passes = (
        improvement >= 0.03
        and bool(cand.get("promotion_eligible"))
        and no_major_regression
    )

    print(f"\nBaseline overall : {baseline['score']:.4f}")
    print(f"Candidate overall: {cand['score']:.4f}")
    print(f"Improvement      : {improvement:+.4f}")
    print(f"Hard gates       : {'PASS' if cand.get('promotion_eligible') else 'FAIL'}")
    print(f"No regressions   : {'PASS' if no_major_regression else 'FAIL'}")

    from .config import MODELS_DIR, DATA_DIR
    candidate_path = MODELS_DIR / candidate["path"]
    active_path = MODELS_DIR / active["path"]
    if candidate_path.resolve() == active_path.resolve():
        raise RuntimeError(
            "Safety stop: candidate path resolves to the active model path. Nothing was deleted."
        )

    if passes:
        canonical = MODELS_DIR / f"butterfly-v{candidate_version}.safetensors"
        if candidate_path.resolve() != canonical.resolve():
            move_model_artifacts(candidate_path, canonical)
            update_entry(candidate_version, path=canonical.name)
        update_entry(
            candidate_version,
            score=cand["score"],
            metadata={
                "benchmark": str(report_path.relative_to(ROOT)),
                "benchmark_suite": BENCHMARK_SUITE_VERSION,
                "storage_format": "safetensors-weights-only",
                "optimizer_included": False,
            },
        )
        promote(candidate_version)
        append_history(
            candidate_version,
            "promoted",
            score=cand["score"],
            metadata={
                "parameters": (candidate.get("metadata") or {}).get("parameters"),
                "tokenizer_vocab": (candidate.get("metadata") or {}).get("tokenizer_vocab"),
                "benchmark": str(report_path.relative_to(ROOT)),
                "benchmark_suite": BENCHMARK_SUITE_VERSION,
                "brain_format": "safetensors-weights-only",
            },
        )
        compact_to_active()
        LEGACY_TOKENIZER_PATH.unlink(missing_ok=True)
        for duplicate in (
            DATA_DIR / "consolidated.txt",
            DATA_DIR / "distilled.txt",
            DATA_DIR / "distilled.jsonl",
        ):
            duplicate.unlink(missing_ok=True)

        print(f"\nPROMOTED Butterfly v{candidate_version}.")
        print("The accepted brain passed BOTH score improvement and strict semantic hard gates.")
        print("Old physical brain/tokenizer are burned only after successful promotion.")
        print("Corpus, memory, verified knowledge, lessons and benchmarks remain.")
        return True

    print("\nREJECTED: candidate failed the current strict promotion policy.")
    if improvement < 0.03:
        print("  - Overall improvement is below +0.0300.")
    if not cand.get("promotion_eligible"):
        for reason in cand.get("promotion_blockers", []):
            print(f"  - {reason}")
    if not no_major_regression:
        print("  - At least one major capability regressed by more than 0.02.")

    append_history(
        candidate_version,
        "rejected",
        score=cand.get("score"),
        metadata={
            "benchmark": str(report_path.relative_to(ROOT)),
            "benchmark_suite": BENCHMARK_SUITE_VERSION,
            "critical_failures": cand.get("critical_failures", []),
        },
    )
    delete_model_artifacts(candidate_path)
    _remove_registry_version(candidate_version)
    print("Candidate brain deleted; corpus, tokenizer assets, memory and report were kept.")
    return False
