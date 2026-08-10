from __future__ import annotations
from pathlib import Path
import hashlib, json, shutil, time, zipfile
from .config import ROOT, MODELS_DIR, ensure_dirs
from .registry import get_active_entry, load_history, resolve_tokenizer_path
from .checkpoint import metadata_path

RELEASE_DIR = ROOT / "release"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_active_release():
    ensure_dirs()
    entry = get_active_entry()
    if not entry:
        raise RuntimeError("No active Butterfly model")
    model_path = MODELS_DIR / entry["path"]
    if model_path.suffix != ".safetensors":
        raise RuntimeError("The active brain is legacy .pt. Promote v0.0004 first; stable v0.0004 uses safetensors weights-only.")
    meta = metadata_path(model_path)
    if not meta.exists():
        raise FileNotFoundError(meta)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    version = entry["version"]
    zip_path = RELEASE_DIR / f"ButterflyAI-brain-v{version}.zip"
    tokenizer_path = resolve_tokenizer_path(entry)
    if not tokenizer_path or not tokenizer_path.exists():
        raise FileNotFoundError("Active Butterfly tokenizer not found; release would not be self-contained.")
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
        "note": "Inference-only ButterflyAI weights + tokenizer. Optimizer/training state intentionally excluded.",
    }
    manifest_path = RELEASE_DIR / f"ButterflyAI-brain-v{version}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(model_path, arcname=model_path.name)
        z.write(meta, arcname=meta.name)
        z.write(tokenizer_path, arcname=tokenizer_path.name)
        z.write(manifest_path, arcname=manifest_path.name)
    print(f"Release package: {zip_path}")
    print(f"Brain: {model_path.stat().st_size/1024/1024:.1f} MB (weights only)")
    print(f"ZIP  : {zip_path.stat().st_size/1024/1024:.1f} MB")
    print("Upload this ZIP as a GitHub Release asset; do not git add it to the repository.")
    return zip_path


def storage_status():
    entry = get_active_entry()
    rows = []
    if entry:
        p = MODELS_DIR / entry["path"]
        rows.append({"active": entry["version"], "path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size if p.exists() else None})
    return {"active_model": rows, "history": load_history(), "release_dir": str(RELEASE_DIR)}
