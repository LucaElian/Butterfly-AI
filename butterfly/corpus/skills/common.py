from __future__ import annotations

import hashlib
import random


def row(user: str, assistant: str, family: str, skill: str) -> dict:
    return {
        "user": user.strip(),
        "assistant": assistant.strip(),
        "family": family,
        "skill": skill,
        "source": "generated",
    }


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def dedupe(rows: list[dict], normalizer) -> list[dict]:
    seen = set()
    out = []
    for item in rows:
        key = (normalizer(item["user"]), item["assistant"].casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def sample(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if not limit or len(rows) <= limit:
        return list(rows)
    return random.Random(seed).sample(rows, int(limit))
