from pathlib import Path
import torch
from torch.utils.data import Dataset
from .tokenizer import ButterflyTokenizer
from .config import TOKENIZER_PATH


class NextTokenDataset(Dataset):
    def __init__(self, text: str, seq_len: int, tokenizer=None):
        self.tokenizer = tokenizer or ButterflyTokenizer.load(TOKENIZER_PATH)
        self.tokens = torch.tensor(self.tokenizer.encode(text, add_bos=True, add_eos=True), dtype=torch.long)
        self.seq_len = seq_len
        if len(self.tokens) < seq_len + 2:
            repeats = (seq_len + 2) // max(1, len(self.tokens)) + 1
            self.tokens = self.tokens.repeat(repeats)

    def __len__(self):
        return max(1, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return x, y


def load_text(path: Path):
    return path.read_text(encoding="utf-8")
