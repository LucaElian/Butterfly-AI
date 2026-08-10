from pathlib import Path
import torch
from butterfly.config import ModelConfig
from butterfly.model import ButterflyTransformer
from butterfly.checkpoint import save_stable_model, load_checkpoint, metadata_path


def test_safetensors_roundtrip(tmp_path: Path):
    cfg = ModelConfig(vocab_size=300, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    model = ButterflyTransformer(cfg)
    path = tmp_path / "brain.safetensors"
    save_stable_model(path, model, extra={"version": "test"})
    loaded, payload = load_checkpoint(path)
    assert metadata_path(path).exists()
    assert payload["extra"]["version"] == "test"
    model.eval(); loaded.eval()
    x = torch.randint(0, 300, (1, 8))
    with torch.no_grad():
        a, _ = model(x)
        b, _ = loaded(x)
    assert torch.allclose(a, b)
