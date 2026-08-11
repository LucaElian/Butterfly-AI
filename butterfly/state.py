from __future__ import annotations

from .experiments import load_current_experiment
from .registry import load_registry, load_history


def system_snapshot() -> dict:
    from .learning.evaluator import BENCHMARK_SUITE_ID
    from .pipeline import pipeline_status

    reg = load_registry()
    return {
        "registry_schema": reg.get("schema_version"),
        "active_brain": reg.get("active"),
        "lab_brain": reg.get("lab"),
        "candidate_brain": reg.get("candidate"),
        "evaluator_suite": BENCHMARK_SUITE_ID,
        "experiment": load_current_experiment(),
        "pipeline": pipeline_status(),
        "history": load_history().get("versions", [])[-12:],
    }
