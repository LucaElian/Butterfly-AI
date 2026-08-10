from dataclasses import dataclass, asdict
from pathlib import Path
import json

APP_VERSION = "0.00041"

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    max_seq_len: int = 384
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.10
    tokenizer_version: int = 3

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in value.items() if k in allowed})

PRESETS = {
    "ryzen3600": ModelConfig(),
    "light": ModelConfig(max_seq_len=320, d_model=320, n_layers=6, n_heads=8, d_ff=1280),
}

def config_for_preset(name: str, vocab_size: int):
    if name == "auto":
        name = "ryzen3600"
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    cfg = ModelConfig(**PRESETS[name].to_dict())
    cfg.vocab_size = vocab_size
    return cfg, name

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STATIC_DIR = DATA_DIR / "static"
CORPUS_DIR = DATA_DIR / "corpus"
MODELS_DIR = ROOT / "models"
STATE_DIR = ROOT / ".butterfly"
TOKENIZERS_DIR = STATE_DIR / "tokenizers"
DB_PATH = STATE_DIR / "butterfly.db"
REGISTRY_PATH = MODELS_DIR / "registry.json"
LEGACY_TOKENIZER_PATH = STATE_DIR / "tokenizer-v2.json"
V4_TOKENIZER_PATH = TOKENIZERS_DIR / "tokenizer-v3-v0.0004.json"
BENCHMARKS_DIR = ROOT / "benchmarks"
INHERITED_DIR = DATA_DIR / "inherited"

LANG_TRAIN = CORPUS_DIR / "language_train.txt"
LANG_VALID = CORPUS_DIR / "language_valid.txt"
CONV_TRAIN = CORPUS_DIR / "conversation_train.txt"
CONV_VALID = CORPUS_DIR / "conversation_valid.txt"
INST_TRAIN = CORPUS_DIR / "instruction_train.txt"
INST_VALID = CORPUS_DIR / "instruction_valid.txt"
WIKI_SOURCES = CORPUS_DIR / "wikipedia_sources.jsonl"
CORPUS_MANIFEST = CORPUS_DIR / "manifest.json"


def ensure_dirs():
    for p in (DATA_DIR, STATIC_DIR, CORPUS_DIR, MODELS_DIR, STATE_DIR, TOKENIZERS_DIR, BENCHMARKS_DIR, INHERITED_DIR):
        p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
