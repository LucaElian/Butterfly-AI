from pathlib import Path
import torch
from torch.utils.data import Dataset


class BlockTokenDataset(Dataset):
    """Non-overlapping next-token blocks.

    v0.0003 shifted one token per sample, which made a multi-million-token corpus
    needlessly expensive to shuffle. v0.0004 walks the corpus in blocks.
    """
    def __init__(self, text: str, seq_len: int, tokenizer):
        ids=tokenizer.encode(text,add_bos=True,add_eos=True)
        self.tokens=torch.tensor(ids,dtype=torch.long)
        self.seq_len=seq_len
        self.n=max(1,(len(self.tokens)-1)//seq_len)
        if len(self.tokens)<seq_len+1:
            repeats=(seq_len+1)//max(1,len(self.tokens))+1
            self.tokens=self.tokens.repeat(repeats); self.n=1

    def __len__(self): return self.n
    def __getitem__(self,idx):
        start=idx*self.seq_len
        x=self.tokens[start:start+self.seq_len]
        y=self.tokens[start+1:start+self.seq_len+1]
        if len(x)<self.seq_len:
            missing=self.seq_len-len(x)
            x=torch.cat([x,self.tokens[:missing]])
            y=torch.cat([y,self.tokens[1:missing+1]])
        return x,y


def load_text(path:Path):
    return Path(path).read_text(encoding="utf-8",errors="ignore")


def load_many(paths):
    return "\n\n".join(load_text(Path(p)) for p in paths if Path(p).exists() and Path(p).stat().st_size)
