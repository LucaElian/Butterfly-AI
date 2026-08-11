from pathlib import Path
import json
import re


def test_tracked_history_contains_no_absolute_benchmark_paths():
    root = Path(__file__).resolve().parents[1]
    path = root / "models" / "history.json"
    if not path.exists():
        return

    history = json.loads(path.read_text(encoding="utf-8"))
    bad = []

    for row in history.get("versions", []):
        metadata = row.get("metadata") or {}
        benchmark = metadata.get("benchmark")
        if not isinstance(benchmark, str):
            continue

        if re.match(r"^[A-Za-z]:[\\/]", benchmark) or benchmark.startswith(("/", "\\")):
            bad.append((row.get("version"), row.get("status"), benchmark))

    assert not bad, "Absolute benchmark paths found in history: " + repr(bad)
