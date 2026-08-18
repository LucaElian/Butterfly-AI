from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from ..memory import MemoryStore
from .skills.common import row


_META_KEYS = {
    "curriculum_node",
    "dynamic_family",
    "family",
    "source",
    "source_url",
    "source_title",
    "context_text",
}


def _stable_id(*parts: str) -> str:
    raw = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _extract_inline_meta(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in ("curriculum_node", "dynamic_family", "family", "source_url"):
        match = re.search(rf"\b{re.escape(key)}\s*[:=]\s*([^\s,;]+)", text)
        if match:
            meta[key] = match.group(1).strip()
    return meta


def _experience_record(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)) and len(raw) >= 7:
        return {
            "id": raw[0],
            "task": raw[1],
            "context": raw[2],
            "actions": raw[3],
            "result": raw[4],
            "lesson": raw[5],
            "quality": raw[6],
        }
    raise TypeError(f"Unsupported experience row: {raw!r}")


def _metadata(exp: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    context = exp.get("context")
    parsed_context = _parse_json(context)
    if isinstance(parsed_context, dict):
        for key in _META_KEYS:
            if parsed_context.get(key) is not None:
                meta[key] = parsed_context[key]
        if parsed_context.get("context") and not meta.get("context_text"):
            meta["context_text"] = parsed_context["context"]
    elif isinstance(context, str):
        meta.update(_extract_inline_meta(context))
        if context.strip():
            meta.setdefault("context_text", context.strip())

    actions = _parse_json(exp.get("actions"))
    action_rows = actions if isinstance(actions, list) else [actions]
    for action in action_rows:
        if isinstance(action, dict):
            for key in _META_KEYS:
                if action.get(key) is not None:
                    meta.setdefault(key, action[key])

    for field in ("task", "lesson", "result"):
        value = exp.get(field)
        if isinstance(value, str):
            meta.update({k: v for k, v in _extract_inline_meta(value).items() if k not in meta})
    return meta


def _target_aliases(focus_target: dict[str, Any]) -> set[str]:
    aliases = {str(value).casefold() for value in focus_target.get("corpus_aliases") or [] if str(value).strip()}
    for key in ("curriculum_node", "dynamic_family", "family"):
        value = focus_target.get(key)
        if value:
            aliases.add(str(value).casefold())
    return aliases


def _matches_target(exp: dict[str, Any], focus_target: dict[str, Any]) -> bool:
    meta = _metadata(exp)
    node = str(focus_target.get("curriculum_node") or "").casefold()
    family = str(focus_target.get("dynamic_family") or focus_target.get("family") or "").casefold()
    exp_node = str(meta.get("curriculum_node") or "").casefold()
    exp_family = str(meta.get("dynamic_family") or meta.get("family") or "").casefold()
    if node and exp_node and node == exp_node:
        return True
    if family and exp_family and family == exp_family:
        return True

    aliases = _target_aliases(focus_target)
    if not aliases:
        return False
    haystack = " ".join(
        str(exp.get(key) or "") for key in ("task", "context", "result", "lesson")
    ).casefold()
    return any(alias in haystack for alias in aliases)


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _training_row(exp: dict[str, Any], focus_target: dict[str, Any], index: int) -> dict | None:
    answer = str(exp.get("lesson") or exp.get("result") or "").strip()
    task = str(exp.get("task") or "").strip()
    if not answer or not task:
        return None

    meta = _metadata(exp)
    node = str(meta.get("curriculum_node") or focus_target.get("curriculum_node") or "general")
    family = str(meta.get("dynamic_family") or meta.get("family") or focus_target.get("dynamic_family") or focus_target.get("family") or "verified")
    source = str(meta.get("source_url") or meta.get("source") or meta.get("source_title") or "verified_experience")
    prompt_parts = [
        f"Material verificado para {node}.",
        f"Tarea: {_clip(task, 260)}",
    ]
    context_text = str(meta.get("context_text") or "").strip()
    if context_text:
        prompt_parts.append(f"Contexto: {_clip(context_text, 360)}")
    prompt_parts.append("Responde con la leccion comprobada, sin inventar datos externos.")

    item = row(
        "\n".join(prompt_parts),
        _clip(answer, 900),
        f"verified_experience:{node}:{_stable_id(str(exp.get('id')), task, answer)}:{index}",
        f"verified_experience:{family}",
    )
    item["source"] = "verified_experience"
    item["provenance"] = {
        "experience_id": exp.get("id"),
        "curriculum_node": node,
        "dynamic_family": family,
        "source": source,
        "quality": exp.get("quality"),
    }
    return item


def build_verified_experience_rows(
    experiences: Iterable[Any],
    focus_target: dict[str, Any],
    *,
    limit: int = 32,
) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    used_ids: list[int] = []
    if not focus_target.get("curriculum_node") and not focus_target.get("dynamic_family"):
        return rows, used_ids
    for raw in experiences:
        exp = _experience_record(raw)
        if not _matches_target(exp, focus_target):
            continue
        item = _training_row(exp, focus_target, len(rows))
        if not item:
            continue
        rows.append(item)
        if exp.get("id") is not None:
            used_ids.append(int(exp["id"]))
        if len(rows) >= int(limit):
            break
    return rows, used_ids


def build_verified_experience_packets(
    focus_target: dict[str, Any],
    *,
    limit: int = 32,
    minimum_quality: float = 0.7,
) -> tuple[list[dict], list[int]]:
    return build_verified_experience_rows(
        MemoryStore().approved_experiences(limit=5000, minimum_quality=minimum_quality),
        focus_target,
        limit=limit,
    )


def mark_verified_experiences_used(ids: Iterable[int]) -> None:
    MemoryStore().mark_used(sorted({int(value) for value in ids}))