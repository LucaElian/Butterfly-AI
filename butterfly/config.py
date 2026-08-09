from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    max_seq_len: int = 384
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.1
    tokenizer_version: int = 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        # Old checkpoints do not have tokenizer_version; dataclass defaults cover it.
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in value.items() if k in allowed})


# Chosen for Luca's Ryzen 5 3600 / 16 GB RAM. RX 580 8 GB is useful for graphics,
# but this Windows/PyTorch build intentionally assumes CPU training for stability.
PRESETS = {
    "ryzen3600": ModelConfig(max_seq_len=384, d_model=384, n_layers=8, n_heads=8, d_ff=1536),
    "light": ModelConfig(max_seq_len=320, d_model=320, n_layers=6, n_heads=8, d_ff=1280),
}


def config_for_preset(name: str, vocab_size: int):
    if name == "auto":
        name = "ryzen3600"
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    base = PRESETS[name]
    cfg = ModelConfig(**base.to_dict())
    cfg.vocab_size = vocab_size
    return cfg, name


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
STATE_DIR = ROOT / ".butterfly"
DB_PATH = STATE_DIR / "butterfly.db"
REGISTRY_PATH = MODELS_DIR / "registry.json"
TOKENIZER_PATH = STATE_DIR / "tokenizer-v2.json"
BENCHMARKS_DIR = ROOT / "benchmarks"
INHERITED_DIR = DATA_DIR / "inherited"


def ensure_dirs():
    for p in (DATA_DIR, MODELS_DIR, STATE_DIR, BENCHMARKS_DIR, INHERITED_DIR):
        p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
