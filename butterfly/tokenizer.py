from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import json
import re


@dataclass
class ButterflyTokenizer:
    """ButterflyAI tokenizer v2.

    It always understands any UTF-8 text through byte fallback, then learns a compact
    vocabulary of frequent words/subwords from Butterfly's accumulated corpus.

    IDs:
      0 BOS, 1 EOS, 2 PAD, 3..258 raw bytes, 259+ learned UTF-8 pieces.
    """

    pieces: list[str]

    BOS = 0
    EOS = 1
    PAD = 2
    BYTE_OFFSET = 3
    LEARNED_OFFSET = 259

    def __post_init__(self):
        self.piece_to_id = {p: self.LEARNED_OFFSET + i for i, p in enumerate(self.pieces)}
        self._first: dict[str, list[str]] = {}
        for p in self.pieces:
            if p:
                self._first.setdefault(p[0], []).append(p)
        for values in self._first.values():
            values.sort(key=len, reverse=True)

    @property
    def vocab_size(self) -> int:
        return self.LEARNED_OFFSET + len(self.pieces)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.BOS)
        i = 0
        while i < len(text):
            match = None
            for candidate in self._first.get(text[i], ()):  # longest-first
                if text.startswith(candidate, i):
                    match = candidate
                    break
            if match is not None:
                ids.append(self.piece_to_id[match])
                i += len(match)
                continue
            raw = text[i].encode("utf-8")
            ids.extend(self.BYTE_OFFSET + b for b in raw)
            i += 1
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids) -> str:
        out = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if self.BYTE_OFFSET <= token_id < self.LEARNED_OFFSET:
                out.append(token_id - self.BYTE_OFFSET)
            elif token_id >= self.LEARNED_OFFSET:
                idx = token_id - self.LEARNED_OFFSET
                if 0 <= idx < len(self.pieces):
                    out.extend(self.pieces[idx].encode("utf-8"))
        return bytes(out).decode("utf-8", errors="replace")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 2, "pieces": self.pieces}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ButterflyTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(list(data.get("pieces", [])))

    @classmethod
    def train(cls, text: str, target_vocab: int = 4096) -> "ButterflyTokenizer":
        # The byte fallback already contributes 259 IDs. We only learn the remainder.
        learned_limit = max(0, target_vocab - cls.LEARNED_OFFSET)
        if learned_limit == 0:
            return cls([])

        word_counts: Counter[str] = Counter()
        sub_counts: Counter[str] = Counter()

        # Preserve a leading space when present so common word boundaries can become one token.
        for m in re.finditer(r"(?:[ \t])?[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", text):
            piece = m.group(0)
            if len(piece) >= 2:
                word_counts[piece] += 1
            core = piece.lstrip()
            weight = 1
            if len(core) >= 4:
                # Frequent inner pieces provide fallback for related/unseen words.
                for n in (3, 4, 5, 6, 7, 8):
                    if len(core) < n:
                        break
                    for i in range(len(core) - n + 1):
                        sub_counts[core[i:i+n]] += weight

        # Useful structural pieces for chat and prose.
        structural = ["\n", "\n\n", ". ", ", ", ": ", "User: ", "Butterfly: ", "<END>\n"]
        selected: list[str] = []
        seen = set()
        for p in structural:
            if p not in seen:
                selected.append(p); seen.add(p)

        ranked_words = sorted(word_counts.items(), key=lambda kv: (kv[1] * max(2, len(kv[0])), kv[1]), reverse=True)
        for p, freq in ranked_words:
            if freq < 2 or p in seen:
                continue
            selected.append(p); seen.add(p)
            if len(selected) >= int(learned_limit * 0.72):
                break

        ranked_sub = sorted(sub_counts.items(), key=lambda kv: (kv[1] * len(kv[0]), kv[1]), reverse=True)
        for p, freq in ranked_sub:
            if freq < 3 or p in seen:
                continue
            selected.append(p); seen.add(p)
            if len(selected) >= learned_limit:
                break

        return cls(selected[:learned_limit])


class ByteTokenizer:
    """Legacy v0.0001/v0.0002 byte tokenizer, kept only for migration."""
    BOS = 256
    EOS = 257
    PAD = 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids):
        raw = bytes(i for i in ids if 0 <= int(i) <= 255)
        return raw.decode("utf-8", errors="replace")
