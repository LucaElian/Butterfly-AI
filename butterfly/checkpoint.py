from __future__ import annotations
from pathlib import Path
import json
import torch
from safetensors.torch import save_model as save_safetensors_model, load_model as load_safetensors_model
from .model import ButterflyTransformer
from .config import ModelConfig, MODELS_DIR, ROOT, LEGACY_TOKENIZER_PATH
from .registry import get_active_entry, resolve_tokenizer_path
from .tokenizer import load_tokenizer


def metadata_path(path: Path) -> Path:
    path = Path(path)
    return path.with_suffix(path.suffix + ".json")


def save_checkpoint(path: Path, model: ButterflyTransformer, optimizer=None, extra=None):
    """Legacy/training checkpoint writer.

    .pt can optionally contain optimizer state and is meant only for temporary/local
    training state. Stable Butterfly brains use save_stable_model() instead.
    """
    path = Path(path)
    payload = {"config": model.cfg.to_dict(), "model": model.state_dict(), "extra": extra or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_stable_model(path: Path, model: ButterflyTransformer, extra=None):
    """Save only inference weights in safetensors + a small JSON sidecar.

    No Adam/optimizer state is stored here. This keeps an accepted Butterfly brain
    close to the theoretical model-weight size instead of ~3x larger.
    """
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("Stable Butterfly models must use .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors_model(model, str(path), metadata={"format": "ButterflyAI", "weights": "inference-only"})
    sidecar = {
        "format": "ButterflyAI-stable-v1",
        "config": model.cfg.to_dict(),
        "extra": extra or {},
    }
    metadata_path(path).write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")


def delete_model_artifacts(path: Path):
    path = Path(path)
    path.unlink(missing_ok=True)
    if path.suffix == ".safetensors":
        metadata_path(path).unlink(missing_ok=True)


def move_model_artifacts(src: Path, dst: Path):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    if src.suffix == ".safetensors":
        metadata_path(dst).unlink(missing_ok=True)
        src.replace(dst)
        src_meta = metadata_path(src)
        if src_meta.exists():
            src_meta.replace(metadata_path(dst))
    else:
        src.replace(dst)


def load_checkpoint(path: Path, device="cpu"):
    path = Path(path)
    if path.suffix == ".safetensors":
        meta_path = metadata_path(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing Butterfly metadata sidecar: {meta_path}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = ModelConfig.from_dict(payload["config"])
        model = ButterflyTransformer(cfg).to(device)
        # safetensors expects its device argument as a string (e.g. "cpu" or
        # "cuda:0").  PyTorch code may hand us torch.device("cpu"), which is
        # valid for model.to(...) but safetensors rejects it with:
        #   SafetensorError: device device(type='cpu') is invalid
        # Normalize here so both fresh loads and stage-recovery loads work.
        safe_device = str(device) if isinstance(device, torch.device) else device
        missing, unexpected = load_safetensors_model(model, str(path), strict=True, device=safe_device)
        if missing or unexpected:
            raise RuntimeError(f"Stable model mismatch. missing={missing} unexpected={unexpected}")
        return model, payload

    # Backward compatibility with v0.0001-v0.0003 .pt brains.
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(payload["config"])
    model = ButterflyTransformer(cfg).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def tokenizer_for_entry(entry):
    p = resolve_tokenizer_path(entry)
    if p and p.exists():
        return load_tokenizer(p)
    if LEGACY_TOKENIZER_PATH.exists():
        return load_tokenizer(LEGACY_TOKENIZER_PATH)
    raise FileNotFoundError("No tokenizer associated with this Butterfly model.")


def load_entry(entry, device="cpu"):
    model, payload = load_checkpoint(MODELS_DIR / entry["path"], device=device)
    return model, payload, tokenizer_for_entry(entry)


def load_active(device="cpu"):
    entry = get_active_entry()
    if not entry:
        raise FileNotFoundError("No active Butterfly model. Train one first.")
    return load_entry(entry, device=device)
