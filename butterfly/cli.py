from __future__ import annotations

import argparse
import json

from .config import ensure_dirs, BENCHMARKS_DIR
from .memory import MemoryStore
from .trainer import best_device
from .checkpoint import load_active
from .generation import generate
from .epistemic.engine import EpistemicEngine
from .agent.preflight import PreflightEvaluator
from .learning.sleep_cycle import run_sleep_cycle
from .learning.evaluator import (
    BENCHMARK_SUITE_VERSION,
    behavior_benchmark,
    print_benchmark,
    save_benchmark,
)

def command_init(_args):
    ensure_dirs()
    MemoryStore()
    print("ButterflyAI permanent installation initialized.")

def command_compare(args):
    from .upgrade import compare_and_promote
    compare_and_promote(args.candidate)

def command_prepare_target(args):
    from .upgrade import prepare_target
    prepare_target(args.target, expected_active=args.expected_active)

def command_chat(args):
    model, _, tokenizer = load_active(device=best_device())
    print("ButterflyAI local chat. Ctrl+C to exit.")
    history = []
    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        context = "".join(
            f"User: {u}\nButterfly: {a}\n<END>\n" for u, a in history[-4:]
        )
        shaped = context + f"User: {prompt}\nButterfly:"
        output = generate(
            model,
            shaped,
            tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        answer = output[len(shaped):]
        for marker in ("<END>", "\nUser:"):
            if marker in answer:
                answer = answer.split(marker, 1)[0]
        answer = answer.strip()
        print("Butterfly >", answer)
        history.append((prompt, answer))

def command_evaluate(_args):
    from .registry import get_active_entry, append_history
    entry = get_active_entry()
    if not entry:
        raise RuntimeError("No active Butterfly model.")
    model, _, tok = load_active(device=best_device())
    metrics = behavior_benchmark(model, tok)
    print_benchmark(metrics)
    path = BENCHMARKS_DIR / (
        f"baseline-v{entry['version']}-suite-v{BENCHMARK_SUITE_VERSION}.json"
    )
    save_benchmark(path, metrics)
    append_history(
        entry["version"],
        f"benchmark-suite-v{BENCHMARK_SUITE_VERSION}",
        score=metrics.get("score"),
        metadata={
            "benchmark": str(path),
            "promotion_eligible": metrics.get("promotion_eligible", False),
            "critical_failures": metrics.get("critical_failures", []),
        },
    )
    print(f"\nStrict baseline saved: {path}")
    print("No model weights, tokenizer, corpus or memory were changed.")

def command_verify(args):
    r = EpistemicEngine().verify(args.claim, allow_web=args.web)
    print(json.dumps({
        "claim": r.claim,
        "status": r.status.value,
        "confidence": r.confidence,
        "method": r.method,
        "explanation": r.explanation,
        "evidence": [e.__dict__ for e in r.evidence],
    }, indent=2, ensure_ascii=False))

def command_preflight(args):
    print(json.dumps(
        PreflightEvaluator().evaluate(args.task).to_dict(),
        indent=2,
        ensure_ascii=False,
    ))

def command_experience(args):
    MemoryStore().add_experience(
        task=args.task,
        result=args.result,
        lesson=args.lesson,
        context=args.context,
        verified=args.verified,
        quality=args.quality,
    )
    print("Experience stored.")

def command_sleep(args):
    run_sleep_cycle(steps=args.steps)

def command_status(_args):
    from .registry import load_registry, load_history
    from .pipeline import load_recipe, load_state
    reg = load_registry()
    try:
        recipe = load_recipe()
        pstate = load_state(recipe)
        pipeline = {
            "target_brain": recipe.get("target_brain"),
            "benchmark_suite": recipe.get("benchmark_suite"),
            "stages": pstate.get("stages"),
            "last_error": pstate.get("last_error"),
        }
    except Exception as exc:
        pipeline = {"error": str(exc)}
    print(json.dumps({
        "pipeline_infrastructure": "1.0",
        "evaluator_suite": BENCHMARK_SUITE_VERSION,
        "active_brain": reg.get("active"),
        "models": reg.get("versions", []),
        "pipeline": pipeline,
        "history": load_history().get("versions", [])[-10:],
    }, indent=2, ensure_ascii=False))

def command_export_release(_args):
    from .storage import export_active_release
    export_active_release()

def command_storage(_args):
    from .storage import storage_status
    print(json.dumps(storage_status(), indent=2, ensure_ascii=False))

def build_parser():
    p = argparse.ArgumentParser(prog="butterfly")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init")
    sp.set_defaults(func=command_init)

    sp = sub.add_parser("prepare-target")
    sp.add_argument("--target", required=True)
    sp.add_argument("--expected-active", default=None)
    sp.set_defaults(func=command_prepare_target)

    sp = sub.add_parser("compare-promote")
    sp.add_argument(
        "--candidate",
        default=None,
        help="Candidate brain version. If omitted, Butterfly selects the sole registered candidate.",
    )
    sp.set_defaults(func=command_compare)

    sp = sub.add_parser("chat")
    sp.add_argument("--max-tokens", type=int, default=150)
    sp.add_argument("--temperature", type=float, default=.58)
    sp.add_argument("--repetition-penalty", type=float, default=1.24)
    sp.set_defaults(func=command_chat)

    sp = sub.add_parser("evaluate")
    sp.set_defaults(func=command_evaluate)

    sp = sub.add_parser("status")
    sp.set_defaults(func=command_status)

    sp = sub.add_parser("storage")
    sp.set_defaults(func=command_storage)

    sp = sub.add_parser("export-release")
    sp.set_defaults(func=command_export_release)

    sp = sub.add_parser("verify")
    sp.add_argument("claim")
    sp.add_argument("--web", action="store_true")
    sp.set_defaults(func=command_verify)

    sp = sub.add_parser("preflight")
    sp.add_argument("task")
    sp.set_defaults(func=command_preflight)

    sp = sub.add_parser("experience")
    sp.add_argument("--task", required=True)
    sp.add_argument("--result", required=True)
    sp.add_argument("--lesson", default="")
    sp.add_argument("--context", default="")
    sp.add_argument("--verified", action="store_true")
    sp.add_argument("--quality", type=float, default=.8)
    sp.set_defaults(func=command_experience)

    sp = sub.add_parser("sleep")
    sp.add_argument("--steps", type=int, default=120)
    sp.set_defaults(func=command_sleep)
    return p

def main():
    args = build_parser().parse_args()
    args.func(args)
