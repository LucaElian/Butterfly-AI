from pathlib import Path
import torch
from .model import ButterflyTransformer
from .config import ModelConfig, MODELS_DIR, TOKENIZER_PATH
from .registry import get_active_entry
from .tokenizer import ButterflyTokenizer


def save_checkpoint(path: Path, model: ButterflyTransformer, optimizer=None, extra=None):
    payload = {
        "config": model.cfg.to_dict(),
        "model": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(payload["config"])
    model = ButterflyTransformer(cfg).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def load_active(device="cpu"):
    entry = get_active_entry()
    if not entry:
        raise FileNotFoundError("No active Butterfly model. Train one first.")
    model, payload = load_checkpoint(MODELS_DIR / entry["path"], device=device)
    tokenizer = ButterflyTokenizer.load(TOKENIZER_PATH)
    return model, payload, tokenizer
