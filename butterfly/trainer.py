from pathlib import Path
import os
import time
import torch
from torch.utils.data import DataLoader
from .config import ModelConfig, MODELS_DIR, TOKENIZER_PATH, ensure_dirs
from .model import ButterflyTransformer
from .data import NextTokenDataset, load_text
from .checkpoint import save_checkpoint
from .registry import register_model
from .tokenizer import ButterflyTokenizer


def best_device():
    # CUDA remains supported automatically if Butterfly moves to an NVIDIA machine.
    return "cuda" if torch.cuda.is_available() else "cpu"


def _configure_cpu():
    if not torch.cuda.is_available():
        threads = max(1, min(12, os.cpu_count() or 6))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, threads // 2)))
        except RuntimeError:
            pass
        return threads
    return None


def train_new(data_path: Path, steps=1400, version="0.0003", cfg=None, lr=3e-4, batch_size=None):
    ensure_dirs()
    tokenizer = ButterflyTokenizer.load(TOKENIZER_PATH)
    cfg = cfg or ModelConfig(vocab_size=tokenizer.vocab_size)
    cfg.vocab_size = tokenizer.vocab_size
    device = best_device()
    cpu_threads = _configure_cpu()
    if batch_size is None:
        batch_size = 4 if device == "cpu" else 10

    text = load_text(data_path)
    seq_len = min(cfg.max_seq_len, 192 if device == "cpu" else cfg.max_seq_len)
    ds = NextTokenDataset(text, seq_len, tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model = ButterflyTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.08)
    model.train()
    started = time.time()
    iterator = iter(loader)
    last_loss = None

    print(f"Device: {device}")
    if cpu_threads:
        print(f"CPU threads: {cpu_threads}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Tokenizer vocab: {tokenizer.vocab_size:,}")
    print(f"Training chars: {len(text):,}")
    print(f"Training tokens: {len(tokenizer.encode(text)):,}")
    print(f"Sequence length: {seq_len} | batch size: {batch_size}")

    for step in range(1, steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
        if step == 1 or step % max(1, steps // 30) == 0:
            elapsed = time.time() - started
            rate = step / max(elapsed, 1e-6)
            eta = (steps - step) / max(rate, 1e-6)
            print(f"step {step:>6}/{steps} | loss {last_loss:.4f} | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    path = MODELS_DIR / f"butterfly-v{version}.pt"
    save_checkpoint(path, model, opt, {
        "train_loss": last_loss, "seconds": time.time()-started, "device": device,
        "steps": steps, "training_data": str(data_path), "tokenizer": str(TOKENIZER_PATH.name),
    })
    register_model(path, version, active=True, metadata={
        "train_loss": last_loss, "steps": steps, "parameters": model.parameter_count(), "tokenizer_vocab": tokenizer.vocab_size,
    })
    print(f"Saved {path}")
    return path


def continue_training(model, text: str, tokenizer, steps=100, lr=1e-4, batch_size=4):
    device = next(model.parameters()).device
    ds = NextTokenDataset(text, min(model.cfg.max_seq_len, 160), tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train(); iterator = iter(loader); last = 0.0
    for _ in range(steps):
        try: x, y = next(iterator)
        except StopIteration: iterator = iter(loader); x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); last = float(loss.item())
    return model, last
