from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

from .wiki import build_wikipedia_corpus
from .conversation import build_conversation_corpus
from ..config import ROOT, DATA_DIR, CORPUS_DIR, INST_TRAIN, INST_VALID, CORPUS_MANIFEST, save_json


def _dialog_chunks(text):
    chunks = text.split("<END>")
    return [chunk.strip() + "\n<END>\n\n" for chunk in chunks if chunk.strip()]


def build_instruction_corpus():
    sources = []
    for path in [
        DATA_DIR / "consolidated.txt",
        DATA_DIR / "bootstrap.txt",
        DATA_DIR / "static" / "instruction_seed.txt",
    ]:
        if path.exists() and path.stat().st_size:
            sources.append(path)

    inherited = DATA_DIR / "inherited"
    if inherited.exists():
        sources += list(inherited.glob("*.jsonl"))

    rows = []
    for path in sources:
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    obj = json.loads(line)
                    user = obj.get("user")
                    assistant = obj.get("assistant")
                    if user and assistant and len(assistant) < 1600:
                        rows.append(f"User: {user}\nButterfly: {assistant}\n<END>\n\n")
                except Exception:
                    pass
        else:
            rows.extend(_dialog_chunks(path.read_text(encoding="utf-8", errors="ignore")))

    if not rows:
        rows = ["User: Hola\nButterfly: ¡Hola! ¿Cómo estás?\n<END>\n\n"]

    with INST_TRAIN.open("w", encoding="utf-8") as train, INST_VALID.open("w", encoding="utf-8") as valid:
        for index, item in enumerate(rows):
            (valid if index % 20 == 0 else train).write(item)

    print(
        f"Instruction corpus: "
        f"{sum(len(item.encode('utf-8')) for item in rows)/1024/1024:.2f} MB | "
        f"examples: {len(rows):,}"
    )
    return sources


def _corpus_id(files: dict) -> str:
    raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "corpus-" + hashlib.sha256(raw).hexdigest()[:12]


def build_all(target_wiki_mb=20.0, conversation_mb=2.0):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    build_wikipedia_corpus(target_mb=target_wiki_mb)
    build_conversation_corpus(target_mb=conversation_mb)
    inst_sources = build_instruction_corpus()

    files = {}
    for path in CORPUS_DIR.glob("*.txt"):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "format": 2,
        "corpus_id": _corpus_id(files),
        "built_at": time.time(),
        "seconds": time.time() - started,
        "files": files,
        "instruction_sources": [str(path.relative_to(ROOT)) for path in inst_sources],
        "notes": [
            "Wikipedia text is language-model pretraining material, not automatically trusted epistemic memory.",
            "Source URLs for Wikipedia articles are retained in wikipedia_sources.jsonl.",
            "Conversation corpus is locally generated from ButterflyAI-authored templates.",
        ],
    }
    save_json(CORPUS_MANIFEST, manifest)
    print(f"Corpus manifest: {CORPUS_MANIFEST}")
    return manifest
