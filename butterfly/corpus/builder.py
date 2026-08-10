from __future__ import annotations
from pathlib import Path
import json, re, time
from .wiki import build_wikipedia_corpus
from .conversation import build_conversation_corpus
from ..config import ROOT,DATA_DIR,CORPUS_DIR,INST_TRAIN,INST_VALID,CORPUS_MANIFEST,save_json


def _dialog_chunks(text):
    chunks=text.split("<END>")
    return [c.strip()+"\n<END>\n\n" for c in chunks if c.strip()]

def build_instruction_corpus():
    sources=[]
    for p in [DATA_DIR/"consolidated.txt", DATA_DIR/"bootstrap.txt", DATA_DIR/"static"/"instruction_seed.txt"]:
        if p.exists() and p.stat().st_size:
            sources.append(p)
    inherited=DATA_DIR/"inherited"
    if inherited.exists(): sources += list(inherited.glob("*.jsonl"))
    rows=[]
    for p in sources:
        if p.suffix.lower()==".jsonl":
            for line in p.read_text(encoding="utf-8",errors="ignore").splitlines():
                try:
                    obj=json.loads(line); u=obj.get("user"); a=obj.get("assistant")
                    if u and a and len(a)<1600: rows.append(f"User: {u}\nButterfly: {a}\n<END>\n\n")
                except Exception: pass
        else:
            rows.extend(_dialog_chunks(p.read_text(encoding="utf-8",errors="ignore")))
    # Always include a clean seed, even if the old generated corpus is unavailable.
    if not rows:
        rows=["User: Hola\nButterfly: ¡Hola! ¿Cómo estás?\n<END>\n\n"]
    with INST_TRAIN.open("w",encoding="utf-8") as ft, INST_VALID.open("w",encoding="utf-8") as fv:
        for i,row in enumerate(rows): (fv if i%20==0 else ft).write(row)
    print(f"Instruction corpus: {sum(len(r.encode('utf-8')) for r in rows)/1024/1024:.2f} MB | examples: {len(rows):,}")
    return sources

def build_all(target_wiki_mb=20.0,conversation_mb=2.0):
    CORPUS_DIR.mkdir(parents=True,exist_ok=True); started=time.time()
    build_wikipedia_corpus(target_mb=target_wiki_mb)
    build_conversation_corpus(target_mb=conversation_mb)
    inst_sources=build_instruction_corpus()
    files={}
    for p in CORPUS_DIR.glob("*.txt"): files[p.name]={"bytes":p.stat().st_size}
    manifest={"version":"0.0004","built_at":time.time(),"seconds":time.time()-started,"files":files,"instruction_sources":[str(p.relative_to(ROOT)) for p in inst_sources],"notes":["Wikipedia text is language-model pretraining material, not automatically trusted epistemic memory.","Source URLs for Wikipedia articles are retained in wikipedia_sources.jsonl.","Conversation corpus is locally generated from ButterflyAI-authored templates."]}
    save_json(CORPUS_MANIFEST,manifest); print(f"Corpus manifest: {CORPUS_MANIFEST}")
    return manifest
