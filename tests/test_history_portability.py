from pathlib import Path
import json
import re

from butterfly.config import ROOT, project_relpath


def _is_portable(value: str) -> bool:
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return False
    if value.startswith(("/", "\\")):
        return False
    if "\\" in value:
        return False
    return True


def test_project_relpath_uses_posix_separators():
    value = project_relpath(ROOT / "benchmarks" / "example.json")
    assert value == "benchmarks/example.json"
    assert _is_portable(value)


def test_tracked_history_benchmark_paths_are_portable():
    path = ROOT / "models" / "history.json"
    if not path.exists():
        return

    history = json.loads(path.read_text(encoding="utf-8"))
    bad = []

    for row in history.get("versions", []):
        metadata = row.get("metadata") or {}
        benchmark = metadata.get("benchmark")
        if isinstance(benchmark, str) and not _is_portable(benchmark):
            bad.append((row.get("version"), row.get("status"), benchmark))

    assert not bad, "Non-portable benchmark paths found in history: " + repr(bad)


def test_runtime_persisted_metadata_paths_are_portable_when_present():
    files = (
        ROOT / "models" / "registry.json",
        ROOT / ".butterfly" / "current_experiment.json",
        ROOT / ".butterfly" / "experiment_history.json",
    )
    path_keys = {"benchmark", "evaluation_report", "baseline_path", "log"}
    bad = []

    def walk(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{where}.{key}"
                if key in path_keys and isinstance(item, str) and not _is_portable(item):
                    bad.append((child, item))
                walk(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")

    for path in files:
        if path.exists():
            walk(json.loads(path.read_text(encoding="utf-8")), path.name)

    assert not bad, "Non-portable runtime paths found: " + repr(bad)
