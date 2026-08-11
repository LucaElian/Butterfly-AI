from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import random
import re
import time

import requests

from ..config import CORPUS_DIR, WIKI_SOURCES

# Batched Wikipedia downloader:
# Instead of requesting individual articles from MediaWiki, fetch Wikipedia rows
# in batches (up to 100 articles per HTTP request) from Hugging Face's dataset
# viewer mirror of the Wikimedia Wikipedia dump. This dramatically reduces the
# number of requests and keeps the original Wikipedia article URL as provenance.
ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.es"
SPLIT = "train"
BATCH_LENGTH = 100
USER_AGENT = "ButterflyAI local educational language-model corpus builder"
NOISY_SECTIONS = re.compile(
    r"\n\s*=+\s*(Referencias|Bibliograf[ií]a|Enlaces externos|V[eé]ase tambi[eé]n)\s*=+.*",
    re.I | re.S,
)


def clean_article(title: str, text: str):
    text = (text or "").replace("\r", "\n")
    text = NOISY_SECTIONS.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 600:
        return None
    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(1, len(text)) < 0.55:
        return None
    return f"<DOC>\nTítulo: {title}\n{text}\n</DOC>\n\n"


def _load_seen(path: Path):
    if not path.exists():
        return set()
    return {
        x.strip()
        for x in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if x.strip()
    }


def _load_state(path: Path):
    if not path.exists():
        return {"total_rows": None, "used_offsets": [], "requests": 0}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj.setdefault("total_rows", None)
        obj.setdefault("used_offsets", [])
        obj.setdefault("requests", 0)
        return obj
    except Exception:
        return {"total_rows": None, "used_offsets": [], "requests": 0}


def _save_state(path: Path, state: dict):
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_rows(session: requests.Session, offset: int, length: int = BATCH_LENGTH):
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": length,
    }
    retries = 0
    while True:
        try:
            r = session.get(ROWS_API, params=params, timeout=60)
            if r.status_code in (429, 503):
                wait = min(60, int(r.headers.get("Retry-After", "4")))
                print(f"Dataset server asks to wait {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            retries += 1
            if retries > 8:
                raise RuntimeError(f"Too many dataset download failures: {exc}") from exc
            wait = min(20, 2 ** min(retries, 4))
            print(f"Network error: {exc}. Retry in {wait}s...")
            time.sleep(wait)


def _choose_offset(total_rows: int, used: set[int]) -> int:
    # Deterministic pseudo-random sampling gives us a broad slice of Wikipedia
    # instead of only the first N article IDs, while remaining resumable.
    max_offset = max(0, total_rows - BATCH_LENGTH)
    if max_offset == 0:
        return 0
    rng = random.Random(0xB0773 + len(used) * 104729)
    for _ in range(5000):
        offset = rng.randrange(0, max_offset + 1)
        if offset not in used:
            return offset
    # Extremely unlikely fallback.
    offset = (len(used) * BATCH_LENGTH) % (max_offset + 1)
    while offset in used:
        offset = (offset + BATCH_LENGTH) % (max_offset + 1)
    return offset


def build_wikipedia_corpus(target_mb: float = 20.0, validation_every: int = 20):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    train = CORPUS_DIR / "language_train.txt"
    valid = CORPUS_DIR / "language_valid.txt"
    seen_path = CORPUS_DIR / "wikipedia_seen.txt"
    state_path = CORPUS_DIR / "hf_wikipedia_state.json"

    seen = _load_seen(seen_path)
    state = _load_state(state_path)
    used_offsets = {int(x) for x in state.get("used_offsets", [])}

    target = int(target_mb * 1024 * 1024)
    current = (train.stat().st_size if train.exists() else 0) + (
        valid.stat().st_size if valid.exists() else 0
    )

    print(f"Spanish Wikipedia corpus target: {target_mb:.1f} MB")
    print(f"Resume point: {current / 1024 / 1024:.2f} MB | articles already seen: {len(seen):,}")
    print("Source mode: Hugging Face dataset-server batches (100 Wikipedia articles/request)")
    if current >= target:
        print("Language corpus already reached target.")
        return train, valid

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})

    # Obtain the row count once. The same response already gives us a valid row,
    # but we keep the bootstrap request separate to make the resume state simple.
    if not state.get("total_rows"):
        bootstrap = _get_rows(session, 0, 1)
        state["total_rows"] = int(bootstrap.get("num_rows_total", 0))
        state["requests"] = int(state.get("requests", 0)) + 1
        _save_state(state_path, state)
        if state["total_rows"] <= 0:
            raise RuntimeError("Dataset server did not report a valid row count.")

    total_rows = int(state["total_rows"])
    started = time.time()
    initial_current = current
    accepted_run = 0
    request_run = 0

    while current < target:
        offset = _choose_offset(total_rows, used_offsets)
        payload = _get_rows(session, offset, BATCH_LENGTH)
        rows = payload.get("rows", [])
        if not rows:
            used_offsets.add(offset)
            state["used_offsets"] = sorted(used_offsets)
            _save_state(state_path, state)
            continue

        with (
            train.open("a", encoding="utf-8") as ft,
            valid.open("a", encoding="utf-8") as fv,
            seen_path.open("a", encoding="utf-8") as fs,
            WIKI_SOURCES.open("a", encoding="utf-8") as fm,
        ):
            for item in rows:
                row = item.get("row", {}) or {}
                pid = str(row.get("id", item.get("row_idx", "")))
                title = str(row.get("title", "")).strip()
                raw = row.get("text", "") or ""
                url = row.get("url")
                if not pid or pid in seen:
                    continue

                # Mark every inspected article as seen so resuming never cycles on
                # rejected/short rows either.
                seen.add(pid)
                fs.write(pid + "\n")
                doc = clean_article(title, raw)
                if not doc:
                    continue

                # Stable held-out split independent of download order.
                try:
                    numeric = int(pid)
                except ValueError:
                    numeric = abs(hash(pid))
                is_valid = numeric % validation_every == 0
                (fv if is_valid else ft).write(doc)

                b = len(doc.encode("utf-8"))
                current += b
                accepted_run += 1
                fm.write(
                    json.dumps(
                        {
                            "id": pid,
                            "title": title,
                            "url": url,
                            "split": "valid" if is_valid else "train",
                            "bytes": b,
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "source": "Spanish Wikipedia via wikimedia/wikipedia dataset mirror",
                            "dataset": DATASET,
                            "dataset_config": CONFIG,
                            "license_note": "Wikipedia text is subject to Wikimedia licensing/attribution; original article URL retained.",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if current >= target:
                    break

        used_offsets.add(offset)
        state["used_offsets"] = sorted(used_offsets)
        state["requests"] = int(state.get("requests", 0)) + 1
        state["last_offset"] = offset
        _save_state(state_path, state)
        request_run += 1

        elapsed = max(0.001, time.time() - started)
        # ETA is calculated only from bytes acquired during this run.
        added = max(1, current - initial_current)
        bytes_per_sec = added / elapsed
        eta = max(0, target - current) / max(1, bytes_per_sec)
        print(
            f"{current / 1024 / 1024:6.2f}/{target_mb:.1f} MB | "
            f"+{accepted_run:,} articles | {request_run} batch requests | ETA ~{eta / 60:.1f}m"
        )
        # Tiny courtesy pause; unlike the old MediaWiki method this is per 100 rows.
        time.sleep(0.20)

    return train, valid
