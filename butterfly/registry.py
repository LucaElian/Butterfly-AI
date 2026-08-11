from __future__ import annotations

from pathlib import Path
import json
import time

from .config import REGISTRY_PATH, MODELS_DIR, ROOT, TOKENIZERS_DIR, ensure_dirs

HISTORY_PATH = MODELS_DIR / "history.json"
REGISTRY_SCHEMA = 2


def _blank_registry():
    return {
        "schema_version": REGISTRY_SCHEMA,
        "active": None,
        "lab": None,
        "candidate": None,
        "versions": [],
    }


def _migrate_registry(value: dict) -> tuple[dict, bool]:
    changed = False
    reg = dict(value or {})
    if reg.get("schema_version") != REGISTRY_SCHEMA:
        changed = True
    reg["schema_version"] = REGISTRY_SCHEMA
    reg.setdefault("active", None)
    reg.setdefault("lab", None)
    reg.setdefault("candidate", None)
    reg.setdefault("versions", [])

    if reg.get("candidate") is None:
        candidates = [
            x.get("version") for x in reg["versions"]
            if x.get("status") == "candidate" and x.get("version") != reg.get("active")
        ]
        if len(candidates) == 1:
            reg["candidate"] = candidates[0]
            changed = True

    if reg.get("lab") is None:
        labs = [x.get("version") for x in reg["versions"] if x.get("status") == "lab"]
        if len(labs) == 1:
            reg["lab"] = labs[0]
            changed = True

    active = reg.get("active")
    for item in reg["versions"]:
        version = item.get("version")
        desired = None
        if version == active:
            desired = "active"
        elif version == reg.get("lab"):
            desired = "lab"
        elif version == reg.get("candidate"):
            desired = "candidate"
        if desired and item.get("status") != desired:
            item["status"] = desired
            changed = True
    return reg, changed


def load_registry():
    ensure_dirs()
    if not REGISTRY_PATH.exists():
        return _blank_registry()
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    reg, changed = _migrate_registry(raw)
    if changed:
        save_registry(reg)
    return reg


def save_registry(reg):
    migrated, _ = _migrate_registry(reg)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_name(REGISTRY_PATH.name + ".tmp")
    tmp.write_text(json.dumps(migrated, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


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
    hist["versions"] = [
        x for x in hist.get("versions", [])
        if not (x.get("version") == version and x.get("status") == status)
    ]
    hist["versions"].append(row)
    HISTORY_PATH.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def get_entry(version: str | None):
    if not version:
        return None
    return next(
        (v for v in load_registry().get("versions", []) if v.get("version") == version),
        None,
    )


def get_active_entry():
    reg = load_registry()
    return get_entry(reg.get("active"))


def get_lab_entry():
    reg = load_registry()
    return get_entry(reg.get("lab"))


def get_candidate_entry():
    reg = load_registry()
    return get_entry(reg.get("candidate"))


def get_seed_entry():
    return get_lab_entry() or get_active_entry()


def register_model(path: Path, version: str, score=None, active=False, metadata=None, status=None):
    reg = load_registry()
    chosen_status = status or ("active" if active else "candidate")
    if chosen_status == "candidate":
        existing = reg.get("candidate")
        if existing and existing != version:
            raise RuntimeError(f"A candidate is already registered: {existing}")
    entry = {
        "version": version,
        "path": str(Path(path).relative_to(MODELS_DIR)),
        "score": score,
        "metadata": metadata or {},
        "status": chosen_status,
    }
    reg["versions"] = [v for v in reg.get("versions", []) if v.get("version") != version]
    reg["versions"].append(entry)
    if chosen_status == "active":
        reg["active"] = version
    elif chosen_status == "lab":
        reg["lab"] = version
    elif chosen_status == "candidate":
        reg["candidate"] = version
    save_registry(reg)
    return entry


def update_entry(version: str, **fields):
    reg = load_registry()
    for item in reg.get("versions", []):
        if item.get("version") == version:
            for key, value in fields.items():
                if key == "metadata":
                    item.setdefault("metadata", {}).update(value)
                else:
                    item[key] = value
            save_registry(reg)
            return item
    raise KeyError(version)


def remove_entry(version: str):
    reg = load_registry()
    reg["versions"] = [x for x in reg.get("versions", []) if x.get("version") != version]
    for slot in ("active", "lab", "candidate"):
        if reg.get(slot) == version:
            reg[slot] = None
    save_registry(reg)


def promote_to_lab(version: str):
    reg = load_registry()
    if reg.get("candidate") != version:
        raise RuntimeError(f"{version} is not the current candidate.")
    old_lab = reg.get("lab")
    found = False
    for item in reg.get("versions", []):
        if item.get("version") == version:
            item["status"] = "lab"
            found = True
        elif item.get("version") == old_lab:
            item["status"] = "superseded"
    if not found:
        raise KeyError(version)
    reg["lab"] = version
    reg["candidate"] = None
    save_registry(reg)
    return old_lab


def promote_to_active(version: str):
    reg = load_registry()
    if version not in {reg.get("candidate"), reg.get("lab")}:
        raise RuntimeError(f"{version} is neither the current candidate nor the current lab.")
    old_active = reg.get("active")
    old_lab = reg.get("lab")
    found = False
    for item in reg.get("versions", []):
        item_version = item.get("version")
        if item_version == version:
            item["status"] = "active"
            found = True
        elif item_version in {old_active, old_lab}:
            item["status"] = "superseded"
    if not found:
        raise KeyError(version)
    reg["active"] = version
    reg["lab"] = None
    reg["candidate"] = None
    save_registry(reg)
    return old_active, old_lab


def resolve_tokenizer_path(entry):
    rel = (entry.get("metadata") or {}).get("tokenizer_path")
    return ROOT / rel if rel else None


def compact_physical_models():
    """Keep only physical ACTIVE/LAB/CANDIDATE brains and their tokenizers."""
    from .checkpoint import delete_model_artifacts

    reg = load_registry()
    keep_versions = {
        value for value in (reg.get("active"), reg.get("lab"), reg.get("candidate")) if value
    }
    keep_entries = []
    tokenizer_keep = set()

    for item in reg.get("versions", []):
        path = MODELS_DIR / item["path"]
        if item.get("version") in keep_versions:
            keep_entries.append(item)
            tp = resolve_tokenizer_path(item)
            if tp:
                tokenizer_keep.add(tp.resolve())
        else:
            delete_model_artifacts(path)

    reg["versions"] = keep_entries
    save_registry(reg)

    for path in TOKENIZERS_DIR.glob("*.json"):
        if path.resolve() not in tokenizer_keep:
            path.unlink(missing_ok=True)
