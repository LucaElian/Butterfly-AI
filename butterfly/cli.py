from __future__ import annotations

import argparse
import json

from .config import ensure_dirs, project_relpath
from .memory import MemoryStore
from .training.runtime import best_device
from .checkpoint import load_active
from .runtime import ButterflyRuntime, load_capabilities, route_deterministic
from .epistemic.engine import EpistemicEngine
from .agent.preflight import PreflightEvaluator
from .learning.sleep_cycle import run_sleep_cycle


def command_init(_args):
    ensure_dirs()
    MemoryStore()
    from .registry import load_registry
    load_registry()
    print("ButterflyAI permanent installation initialized.")


def command_chat(args):
    model, _, tokenizer = load_active(device=best_device())
    runtime = ButterflyRuntime(model, tokenizer)
    print("ButterflyAI unified local chat. Ctrl+C to exit.")
    history = []
    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        response = runtime.respond(
            prompt,
            history,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        print("Butterfly >", response.answer)
        history.append((prompt, response.answer))

def command_evaluate(_args):
    from .registry import get_active_entry, append_history
    from .upgrade import strict_baseline
    from .learning.evaluator import BENCHMARK_SUITE_ID, print_benchmark

    entry = get_active_entry()
    if not entry:
        raise RuntimeError("No ACTIVE Butterfly model.")
    metrics, path = strict_baseline(entry, force=True)
    print_benchmark(metrics)
    append_history(
        entry["version"],
        f"benchmark-suite-{BENCHMARK_SUITE_ID}",
        score=metrics.get("score"),
        metadata={
            "benchmark": project_relpath(path),
            "promotion_eligible": metrics.get("promotion_eligible", False),
            "critical_failures": metrics.get("critical_failures", []),
            "suite_id": BENCHMARK_SUITE_ID,
        },
    )
    print(f"\nStrict baseline saved: {path}")
    print("No model weights, tokenizer, corpus or memory were changed.")


def command_status(_args):
    from .state import system_snapshot
    print(json.dumps(system_snapshot(), indent=2, ensure_ascii=False))


def command_storage(_args):
    from .storage import storage_status
    print(json.dumps(storage_status(), indent=2, ensure_ascii=False))


def command_export_release(_args):
    from .storage import export_active_release
    export_active_release()


def command_verify(args):
    result = EpistemicEngine().verify(args.claim, allow_web=args.web)
    print(json.dumps({
        "claim": result.claim,
        "status": result.status.value,
        "confidence": result.confidence,
        "method": result.method,
        "explanation": result.explanation,
        "evidence": [e.__dict__ for e in result.evidence],
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


def command_experiment_new(args):
    from .experiments import clear_terminal_experiment, create_experiment, load_current_experiment
    current = load_current_experiment()
    if current:
        clear_terminal_experiment()
    exp = create_experiment(args.recipe)
    print(json.dumps(exp, indent=2, ensure_ascii=False))


def command_audit(_args):
    from .audit import print_audit
    raise SystemExit(print_audit())


def command_night_study(args):
    from .learning.night_study import run_night_study
    report = run_night_study(
        max_blocks=args.max_blocks,
        max_minutes=args.max_minutes,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(json.dumps({
            "session_id": report.get("session_id"),
            "stop_reason": report.get("stop_reason"),
            "blocks": len(report.get("blocks", [])),
            "log_path": report.get("log_path"),
        }, indent=2, ensure_ascii=False))


def command_night_plan(_args):
    from .learning.night_study import capability_snapshot, print_plan
    print_plan(capability_snapshot(force_baseline=False))


def command_route(args):
    result = route_deterministic(args.prompt)
    payload = {
        "prompt": args.prompt,
        "route": result.route if result else "neural",
        "deterministic": bool(result),
        "answer": result.answer if result else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def command_capabilities(_args):
    print(json.dumps(load_capabilities(), indent=2, ensure_ascii=False))


def command_health(_args):
    from .learning.evaluator import BENCHMARK_SUITE_ID
    from .registry import load_registry
    reg = load_registry()
    print("ButterflyAI health check: OK")
    print(f"Registry schema : {reg.get('schema_version')}")
    print(f"Evaluator suite : {BENCHMARK_SUITE_ID}")
    print(f"ACTIVE          : {reg.get('active')}")
    print(f"LAB             : {reg.get('lab')}")
    print(f"CANDIDATE       : {reg.get('candidate')}")


def build_parser():
    parser = argparse.ArgumentParser(prog="butterfly")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init")
    sp.set_defaults(func=command_init)

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

    sp = sub.add_parser("experiment-new")
    sp.add_argument("--recipe", default=None)
    sp.set_defaults(func=command_experiment_new)

    sp = sub.add_parser("audit-hardcodes")
    sp.set_defaults(func=command_audit)

    sp = sub.add_parser("health")
    sp.set_defaults(func=command_health)

    sp = sub.add_parser("route")
    sp.add_argument("prompt")
    sp.set_defaults(func=command_route)

    sp = sub.add_parser("capabilities")
    sp.set_defaults(func=command_capabilities)

    sp = sub.add_parser("night-plan")
    sp.set_defaults(func=command_night_plan)

    sp = sub.add_parser("night-study")
    sp.add_argument("--max-blocks", type=int, default=None)
    sp.add_argument("--max-minutes", type=float, default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=command_night_study)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)
