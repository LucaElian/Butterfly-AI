from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader

from ..config import load_autonomy_config
from ..data import BlockTokenDataset


def best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_cpu():
    if torch.cuda.is_available():
        return None
    cfg = load_autonomy_config()
    runtime = cfg.get("runtime") or {}
    fraction = float(runtime.get("cpu_thread_fraction", 0.67))
    fraction = min(1.0, max(0.10, fraction))
    logical = max(1, os.cpu_count() or 1)
    threads = max(1, min(logical, round(logical * fraction)))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads // 2 or 1)))
    except RuntimeError:
        pass
    return threads


def continue_training(model, text, tokenizer, steps=120, lr=8e-5, batch_size=4):
    device = next(model.parameters()).device
    ds = BlockTokenDataset(text, min(model.cfg.max_seq_len, 160), tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.1)
    model.train()
    it = iter(loader)
    last = 0.0
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    return model, last
