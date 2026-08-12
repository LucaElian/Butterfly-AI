from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time
import zipfile

from .config import ROOT, MODELS_DIR, ensure_dirs
from .registry import (
    get_active_entry, get_lab_entry, get_candidate_entry,
    load_history, resolve_tokenizer_path,
)

RELEASE_DIR = ROOT / "release"


def metadata_path(path: Path) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_active_release():
    ensure_dirs()
    entry = get_active_entry()
    if not entry:
        raise RuntimeError("No ACTIVE Butterfly model.")
    model_path = MODELS_DIR / entry["path"]
    if model_path.suffix != ".safetensors":
        raise RuntimeError("The ACTIVE brain uses a legacy format. Export requires stable safetensors weights.")
    meta = metadata_path(model_path)
    if not meta.exists():
        raise FileNotFoundError(meta)

    tokenizer_path = resolve_tokenizer_path(entry)
    if not tokenizer_path or not tokenizer_path.exists():
        raise FileNotFoundError("ACTIVE tokenizer not found; release would not be self-contained.")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    version = entry["version"]
    zip_path = RELEASE_DIR / f"ButterflyAI-brain-v{version}.zip"
    manifest_path = RELEASE_DIR / f"ButterflyAI-brain-v{version}-manifest.json"
    manifest = {
        "version": version,
        "brain_file": model_path.name,
        "metadata_file": meta.name,
        "tokenizer_file": tokenizer_path.name,
        "brain_bytes": model_path.stat().st_size,
        "brain_sha256": sha256(model_path),
        "metadata_sha256": sha256(meta),
        "tokenizer_sha256": sha256(tokenizer_path),
        "registry_metadata": entry.get("metadata") or {},
        "score": entry.get("score"),
        "exported_at": time.time(),
        "note": "Inference-only ButterflyAI weights + tokenizer. Training state excluded.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(model_path, arcname=model_path.name)
        archive.write(meta, arcname=meta.name)
        archive.write(tokenizer_path, arcname=tokenizer_path.name)
        archive.write(manifest_path, arcname=manifest_path.name)
    print(f"Release package: {zip_path}")
    return zip_path


def _slot_row(entry):
    if not entry:
        return None
    path = MODELS_DIR / entry["path"]
    return {
        "version": entry["version"],
        "status": entry.get("status"),
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def storage_status():
    return {
        "active": _slot_row(get_active_entry()),
        "lab": _slot_row(get_lab_entry()),
        "candidate": _slot_row(get_candidate_entry()),
        "history": load_history(),
        "release_dir": str(RELEASE_DIR),
    }
