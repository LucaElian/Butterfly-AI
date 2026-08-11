from __future__ import annotations

from pathlib import Path
import json
import torch
from torch.utils.data import Dataset


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        user = str(obj.get("user", "")).strip()
        assistant = str(obj.get("assistant", "")).strip()
        if not user or not assistant:
            raise ValueError(f"Invalid dialogue row at {path}:{line_no}")
        rows.append({
            "user": user,
            "assistant": assistant,
            "category": str(obj.get("category", "unknown")),
        })
    return rows


class AssistantOnlyDialogueDataset(Dataset):
    """Causal dialogue dataset where loss is computed only on Butterfly's answer."""

    def __init__(self, rows: list[dict], tokenizer, seq_len: int = 192):
        self.rows = rows
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.answer_tokens = 0
        self.truncated = 0

        for row in rows:
            prefix = f"User: {row['user']}\nButterfly:"
            suffix = f" {row['assistant']}\n<END>\n"
            prefix_ids = tokenizer.encode(prefix, add_bos=True)
            suffix_ids = tokenizer.encode(suffix, add_eos=True)
            full = prefix_ids + suffix_ids

            if len(full) > seq_len + 1:
                max_ids = seq_len + 1
                answer_keep = min(len(suffix_ids), max_ids - 1)
                prefix_keep = max_ids - answer_keep
                prefix_ids = prefix_ids[-prefix_keep:]
                suffix_ids = suffix_ids[-answer_keep:]
                full = prefix_ids + suffix_ids
                self.truncated += 1

            x_ids = full[:-1]
            labels = list(full[1:])
            assistant_start = max(0, len(prefix_ids) - 1)
            for index in range(min(assistant_start, len(labels))):
                labels[index] = -100

            self.answer_tokens += sum(1 for value in labels if value != -100)

            if len(x_ids) < seq_len:
                pad = seq_len - len(x_ids)
                x_ids += [tokenizer.PAD] * pad
                labels += [-100] * pad

            self.items.append((
                torch.tensor(x_ids[:seq_len], dtype=torch.long),
                torch.tensor(labels[:seq_len], dtype=torch.long),
            ))

        if not self.items:
            raise ValueError("AssistantOnlyDialogueDataset received no valid rows")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
