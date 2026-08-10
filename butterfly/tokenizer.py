from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import json,re

@dataclass
class LegacyButterflyTokenizerV2:
    pieces:list[str]
    BOS=0;EOS=1;PAD=2;BYTE_OFFSET=3;LEARNED_OFFSET=259
    def __post_init__(self):
        self.piece_to_id={p:self.LEARNED_OFFSET+i for i,p in enumerate(self.pieces)};self._first={}
        for p in self.pieces:
            if p:self._first.setdefault(p[0],[]).append(p)
        for v in self._first.values():v.sort(key=len,reverse=True)
    @property
    def vocab_size(self):return self.LEARNED_OFFSET+len(self.pieces)
    def encode(self,text,add_bos=False,add_eos=False):
        ids=[self.BOS] if add_bos else [];i=0
        while i<len(text):
            match=None
            for candidate in self._first.get(text[i],()):
                if text.startswith(candidate,i):match=candidate;break
            if match is not None:ids.append(self.piece_to_id[match]);i+=len(match);continue
            ids.extend(self.BYTE_OFFSET+b for b in text[i].encode("utf-8"));i+=1
        if add_eos:ids.append(self.EOS)
        return ids
    def decode(self,ids):
        out=bytearray()
        for token_id in ids:
            token_id=int(token_id)
            if self.BYTE_OFFSET<=token_id<self.LEARNED_OFFSET:out.append(token_id-self.BYTE_OFFSET)
            elif token_id>=self.LEARNED_OFFSET:
                idx=token_id-self.LEARNED_OFFSET
                if 0<=idx<len(self.pieces):out.extend(self.pieces[idx].encode("utf-8"))
        return bytes(out).decode("utf-8",errors="replace")

@dataclass
class ButterflySubwordTokenizerV3:
    """Butterfly's self-contained tokenizer v3.

    It keeps guaranteed UTF-8 byte fallback, but learns ~8k frequent whole-word and
    subword pieces from Butterfly's multi-million-token corpus. Encoding uses a trie,
    so an 8k vocabulary stays fast without a third-party tokenizer runtime.
    """
    pieces:list[str]
    BOS=0;EOS=1;PAD=2;BYTE_OFFSET=3;LEARNED_OFFSET=259;FORMAT="butterfly-subword-v3"
    def __post_init__(self):
        self.piece_to_id={p:self.LEARNED_OFFSET+i for i,p in enumerate(self.pieces)}
        self.trie={}
        for p,tid in self.piece_to_id.items():
            node=self.trie
            for ch in p:node=node.setdefault(ch,{})
            node[None]=tid
    @property
    def vocab_size(self):return self.LEARNED_OFFSET+len(self.pieces)
    def encode(self,text,add_bos=False,add_eos=False):
        ids=[self.BOS] if add_bos else [];i=0;n=len(text)
        while i<n:
            node=self.trie.get(text[i]);best_id=None;best_j=i
            if node is not None:
                j=i+1
                if None in node:best_id=node[None];best_j=j
                while j<n and text[j] in node:
                    node=node[text[j]];j+=1
                    if None in node:best_id=node[None];best_j=j
            if best_id is not None:
                ids.append(best_id);i=best_j;continue
            raw=text[i].encode("utf-8");ids.extend(self.BYTE_OFFSET+b for b in raw);i+=1
        if add_eos:ids.append(self.EOS)
        return ids
    def decode(self,ids):
        out=bytearray()
        for token_id in ids:
            token_id=int(token_id)
            if self.BYTE_OFFSET<=token_id<self.LEARNED_OFFSET:out.append(token_id-self.BYTE_OFFSET)
            elif token_id>=self.LEARNED_OFFSET:
                idx=token_id-self.LEARNED_OFFSET
                if 0<=idx<len(self.pieces):out.extend(self.pieces[idx].encode("utf-8"))
        return bytes(out).decode("utf-8",errors="replace")
    def save(self,path:Path):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({"format":self.FORMAT,"version":3,"pieces":self.pieces},ensure_ascii=False),encoding="utf-8")
    @classmethod
    def train(cls,files:list[Path],target_vocab=8192):
        learned_limit=max(0,target_vocab-cls.LEARNED_OFFSET);word_counts=Counter()
        pattern=re.compile(r"(?:[ \t])?[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")
        total_chars=0
        for path in files:
            text=Path(path).read_text(encoding="utf-8",errors="ignore");total_chars+=len(text)
            word_counts.update(m.group(0) for m in pattern.finditer(text) if len(m.group(0).strip())>=2)
        structural=["\n","\n\n",". ",", ",": ","; ","User: ","Butterfly: ","<END>\n","<DOC>\n","</DOC>\n","Título: "]
        selected=[];seen=set()
        for p in structural:
            if p not in seen:selected.append(p);seen.add(p)
        # Whole words are the biggest compression win. Score by both frequency and length.
        ranked=sorted(word_counts.items(),key=lambda kv:(kv[1]*max(2,len(kv[0])),kv[1]),reverse=True)
        whole_budget=int(learned_limit*.66)
        for p,freq in ranked:
            if freq<2 or p in seen:continue
            selected.append(p);seen.add(p)
            if len(selected)>=whole_budget:break
        # Learn reusable prefixes/suffixes/internal pieces from the most useful word types.
        sub=Counter()
        for piece,freq in ranked[:35000]:
            core=piece.lstrip()
            if len(core)<4:continue
            weight=min(freq,200)
            for n in (3,4,5,6,7,8,10,12):
                if len(core)<n:break
                sub[core[:n]]+=weight*2;sub[core[-n:]]+=weight*2
            if freq>=3:
                for n in (3,4,5):
                    if len(core)<n:break
                    # cap very long words to avoid candidate explosion
                    for i in range(min(len(core)-n+1,24)):sub[core[i:i+n]]+=weight
        ranked_sub=sorted(sub.items(),key=lambda kv:(kv[1]*len(kv[0]),kv[1]),reverse=True)
        for p,freq in ranked_sub:
            if freq<4 or p in seen:continue
            selected.append(p);seen.add(p)
            if len(selected)>=learned_limit:break
        print(f"Tokenizer v3 scan: {len(word_counts):,} unique word forms from {total_chars:,} chars; learned {len(selected):,} pieces")
        return cls(selected[:learned_limit])

def load_tokenizer(path:Path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format")==ButterflySubwordTokenizerV3.FORMAT:return ButterflySubwordTokenizerV3(list(data.get("pieces",[])))
    if "pieces" in data:return LegacyButterflyTokenizerV2(list(data.get("pieces",[])))
    raise ValueError(f"Unknown tokenizer format: {path}")

class ButterflyTokenizer:
    @classmethod
    def load(cls,path):return load_tokenizer(Path(path))
    @classmethod
    def train_subword(cls,files,vocab_size=8192):return ButterflySubwordTokenizerV3.train([Path(p) for p in files],target_vocab=vocab_size)
    @classmethod
    def train_bpe(cls,files,vocab_size=8192,min_frequency=2):
        del min_frequency
        return cls.train_subword(files,vocab_size)

class ByteTokenizer:
    BOS=256;EOS=257;PAD=258;vocab_size=259
    def encode(self,text,add_bos=False,add_eos=False):
        ids=list(text.encode("utf-8"));
        if add_bos:ids.insert(0,self.BOS)
        if add_eos:ids.append(self.EOS)
        return ids
    def decode(self,ids):return bytes(i for i in ids if 0<=int(i)<=255).decode("utf-8",errors="replace")
