from __future__ import annotations
from pathlib import Path
import json, time
from .config import REGISTRY_PATH, MODELS_DIR, ROOT, ensure_dirs

HISTORY_PATH = MODELS_DIR / "history.json"


def load_registry():
    ensure_dirs()
    if not REGISTRY_PATH.exists():
        return {"active": None, "versions": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(reg):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def load_history():
    ensure_dirs()
    if not HISTORY_PATH.exists():
        return {"format": 1, "versions": []}
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def append_history(version: str, status: str, score=None, metadata=None):
    hist = load_history()
    row = {
        "version": version,
        "status": status,
        "score": score,
        "recorded_at": time.time(),
        "metadata": metadata or {},
    }
    # One final record per version/status pair; preserve prior milestones compactly.
    hist["versions"] = [x for x in hist.get("versions", []) if not (x.get("version") == version and x.get("status") == status)]
    hist["versions"].append(row)
    HISTORY_PATH.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def register_model(path: Path, version: str, score=None, active=False, metadata=None, status=None):
    reg = load_registry()
    entry = {
        "version": version,
        "path": str(path.relative_to(MODELS_DIR)),
        "score": score,
        "metadata": metadata or {},
        "status": status or ("active" if active else "candidate"),
    }
    reg["versions"] = [v for v in reg.get("versions", []) if v["version"] != version]
    reg["versions"].append(entry)
    if active:
        reg["active"] = version
    save_registry(reg)
    return entry


def get_entry(version: str):
    return next((v for v in load_registry().get("versions", []) if v["version"] == version), None)


def get_active_entry():
    reg = load_registry()
    active = reg.get("active")
    return get_entry(active) if active else None


def get_candidate_entry(version="0.0004"):
    return get_entry(version)


def update_entry(version: str, **fields):
    reg = load_registry()
    for item in reg.get("versions", []):
        if item["version"] == version:
            for k, v in fields.items():
                if k == "metadata":
                    item["metadata"].update(v)
                else:
                    item[k] = v
            save_registry(reg)
            return item
    raise KeyError(version)


def promote(version: str):
    reg = load_registry()
    found = False
    for item in reg.get("versions", []):
        if item["version"] == version:
            item["status"] = "active"
            found = True
        elif item["version"] == reg.get("active"):
            item["status"] = "superseded"
    if not found:
        raise KeyError(version)
    reg["active"] = version
    save_registry(reg)


def resolve_tokenizer_path(entry):
    rel = (entry.get("metadata") or {}).get("tokenizer_path")
    return ROOT / rel if rel else None


def compact_to_active():
    """Keep one accepted physical brain. Memories/corpora/benchmarks are never removed."""
    from .checkpoint import delete_model_artifacts
    reg = load_registry()
    active = reg.get("active")
    if not active:
        return
    keep = []
    tokenizer_keep = set()
    for item in reg.get("versions", []):
        p = MODELS_DIR / item["path"]
        if item["version"] == active:
            keep.append(item)
            tp = resolve_tokenizer_path(item)
            if tp:
                tokenizer_keep.add(tp.resolve())
        else:
            delete_model_artifacts(p)
    reg["versions"] = keep
    save_registry(reg)
    from .config import TOKENIZERS_DIR
    for p in TOKENIZERS_DIR.glob("*.json"):
        if p.resolve() not in tokenizer_keep:
            p.unlink(missing_ok=True)
