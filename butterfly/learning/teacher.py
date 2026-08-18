from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from ..config import CONFIG_DIR, load_json
from ..memory import MemoryStore
from .curriculum_graph import choose_next, load_seed, mark_material


CONFIG_PATH = CONFIG_DIR / "teacher.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "enabled": True,
    "provider": "ollama",
    "model": "qwen2.5:3b",
    "temperature": 0.35,
    "quality": 0.86,
    "limits": {
        "targets_per_run": 1,
        "lessons_per_target": 6,
        "max_lesson_chars": 720,
        "retry_attempts": 2,
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "timeout_seconds": 120,
    },
}

SUPPORTED_TARGETS: dict[str, dict[str, str]] = {
    "instruction.sentence": {"dynamic_family": "sentence", "title": "single-sentence constraints"},
    "instruction.two_steps": {"dynamic_family": "two_steps", "title": "two-step ordered instructions"},
    "instruction.missing": {"dynamic_family": "missing", "title": "detecting missing information"},
    "instruction.short": {"dynamic_family": "short", "title": "concise response constraints"},
    "computing.files": {"dynamic_family": "file", "title": "files"},
    "computing.folders": {"dynamic_family": "folder", "title": "folders and directories"},
    "computing.cpu": {"dynamic_family": "cpu", "title": "CPU and execution"},
    "computing.memory": {"dynamic_family": "ram", "title": "computer memory"},
    "programming.api_basics": {"dynamic_family": "api", "title": "API fundamentals"},
    "programming.functions": {"dynamic_family": "parameter", "title": "functions and parameters"},
    "ai.parameters": {"dynamic_family": "parameter", "title": "model parameters"},
    "ai.tokens": {"dynamic_family": "token", "title": "tokens and tokenization"},
    "ai.datasets": {"dynamic_family": "dataset", "title": "datasets"},
    "ai.training_epochs": {"dynamic_family": "epoch", "title": "epochs in model training"},
}

FAMILY_TO_NODE = {
    value["dynamic_family"]: node_id for node_id, value in SUPPORTED_TARGETS.items()
}


@dataclass(frozen=True)
class TeacherTarget:
    curriculum_node: str
    dynamic_family: str
    title: str


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_teacher_config(path: Path | None = None) -> dict[str, Any]:
    raw = load_json(path or CONFIG_PATH, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RuntimeError("config/teacher.json must contain a JSON object.")
    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    limits = cfg.setdefault("limits", {})
    limits["targets_per_run"] = max(1, int(limits.get("targets_per_run", 1)))
    limits["lessons_per_target"] = max(1, min(24, int(limits.get("lessons_per_target", 6))))
    limits["max_lesson_chars"] = max(160, min(1600, int(limits.get("max_lesson_chars", 720))))
    limits["retry_attempts"] = max(1, min(5, int(limits.get("retry_attempts", 2))))
    cfg["quality"] = max(0.0, min(1.0, float(cfg.get("quality", 0.86))))
    cfg["temperature"] = max(0.0, min(1.0, float(cfg.get("temperature", 0.35))))
    return cfg


def _seed_titles() -> dict[str, str]:
    return {str(row["id"]): str(row.get("title") or row["id"]) for row in load_seed().get("nodes") or []}


def _target(node_id: str) -> TeacherTarget | None:
    spec = SUPPORTED_TARGETS.get(str(node_id))
    if not spec:
        return None
    titles = _seed_titles()
    return TeacherTarget(
        curriculum_node=str(node_id),
        dynamic_family=spec["dynamic_family"],
        title=titles.get(str(node_id), spec["title"]),
    )


def select_teacher_targets(
    *,
    node: str | None = None,
    dynamic_family: str | None = None,
    limit: int = 1,
) -> list[TeacherTarget]:
    if node:
        target = _target(str(node))
        if target is None:
            raise KeyError(f"Unsupported teacher curriculum node: {node}")
        return [target]
    if dynamic_family:
        mapped = FAMILY_TO_NODE.get(str(dynamic_family))
        if not mapped:
            raise KeyError(f"Unsupported teacher dynamic family: {dynamic_family}")
        target = _target(mapped)
        return [target] if target else []

    targets: list[TeacherTarget] = []
    for row in choose_next(require_material=False, limit=max(12, int(limit) * 4)):
        target = _target(str(row.get("id")))
        if target is not None:
            targets.append(target)
        if len(targets) >= int(limit):
            break
    if not targets:
        fallback = _target("instruction.missing")
        if fallback is not None:
            targets.append(fallback)
    return targets[: int(limit)]


def _family_instruction(target: TeacherTarget) -> str:
    rules = {
        "two_steps": (
            "Cada task debe pedir exactamente dos pasos numerados para una accion concreta, sin repetir la consigna. "
            "Cada result debe responder esa accion directamente con solo 1. y 2.; no agregues introduccion, tercer paso, cierre, ni respuestas meta como listar ingredientes o describir como hacerlo. "
            "Prohibido decir que las instrucciones deben ser identicas, duplicadas o repetidas."
        ),
        "missing": (
            "Cada task debe tener una instruccion incompleta. "
            "Cada result debe pedir el dato faltante sin inventarlo."
        ),
        "short": "Cada result debe ser breve y directo, sin relleno.",
        "sentence": "Cada result debe ser una sola oracion clara.",
    }
    return rules.get(target.dynamic_family, "Mantene task, lesson y result concretos, evaluables y no enciclopedicos.")


def _prompt(target: TeacherTarget, count: int) -> str:
    return (
        "Genera material de estudio para un modelo chico llamado ButterflyAI. "
        "No copies benchmarks ni preguntas exactas conocidas. Crea variaciones nuevas, cortas y entrenables. "
        "Responde SOLO JSON valido, sin markdown.\n"
        "Formato: {\"lessons\":[{\"task\":str,\"lesson\":str,\"result\":str}]}\n"
        f"Cantidad: {int(count)}\n"
        f"curriculum_node fijo: {target.curriculum_node}\n"
        f"dynamic_family fijo: {target.dynamic_family}\n"
        f"Tema: {target.title}\n"
        f"Regla de familia: {_family_instruction(target)}\n"
        "Reglas: task debe ser una consigna de practica unica; lesson debe explicar el patron general; "
        "result debe ser una respuesta modelo breve que conteste la task directamente. "
        "No repitas la misma frase dentro de task, lesson o result. Usa espanol claro y ASCII basico."
    )


def _ollama_generate(prompt: str, cfg: dict[str, Any]) -> str:
    ollama_cfg = cfg.get("ollama") if isinstance(cfg.get("ollama"), dict) else {}
    base_url = str(ollama_cfg.get("base_url") or DEFAULT_CONFIG["ollama"]["base_url"]).rstrip("/")
    timeout = float(ollama_cfg.get("timeout_seconds") or DEFAULT_CONFIG["ollama"]["timeout_seconds"])
    model = str(cfg.get("model") or DEFAULT_CONFIG["model"])
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(cfg.get("temperature", 0.35)),
            "num_predict": 1400,
        },
    }
    request = urllib.request.Request(
        base_url + "/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama is not reachable. Open Ollama, run `ollama pull "
            f"{model}`, then run TEACHER_LESSONS.bat again."
        ) from exc
    text = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Ollama returned an empty teacher response.")
    return text


def _extract_json(text: str) -> Any:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(1))


def parse_teacher_lessons(text: str, target: TeacherTarget, *, max_lesson_chars: int = 720) -> list[dict[str, str]]:
    payload = _extract_json(text)
    items = payload.get("lessons") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise RuntimeError("Teacher response must contain a lessons array.")

    from .evaluator import benchmark_surface_prompts, normalize_surface
    from .study_exam import study_surface_prompts

    held_out = set(benchmark_surface_prompts())
    held_out |= {normalize_surface(prompt) for prompt in study_surface_prompts()}
    lessons: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        task = _clean(item.get("task"))
        lesson = _clean(item.get("lesson"))
        result = _clean(item.get("result")) or lesson
        if len(task) < 12 or len(lesson) < 24:
            continue
        if not _valid_teacher_lesson(target, task, lesson, result):
            continue
        if normalize_surface(task) in held_out:
            continue
        signature = _stable_id(target.curriculum_node, target.dynamic_family, task, lesson)
        if signature in seen:
            continue
        seen.add(signature)
        lessons.append({
            "task": task[:360],
            "lesson": lesson[: int(max_lesson_chars)],
            "result": result[:360],
            "curriculum_node": target.curriculum_node,
            "dynamic_family": target.dynamic_family,
            "signature": signature,
        })
    return lessons


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _plain_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _looks_repeated(value: str) -> bool:
    words = _plain_words(value)
    if len(words) < 6 or len(words) % 2:
        return False
    midpoint = len(words) // 2
    return words[:midpoint] == words[midpoint:]


def _contains_bad_teacher_pattern(*values: str) -> bool:
    text = " ".join(values).lower()
    bad_patterns = (
        "identic",
        "duplicad",
        "repetid",
        "misma instruccion",
        "mismas instrucciones",
        "listar los ingredientes",
        "describir como",
        "describir cómo",
    )
    return any(pattern in text for pattern in bad_patterns)


def _numbered_steps(result: str) -> list[str]:
    return re.findall(r"(?:^|\s)(\d+)[\.)]", result)


def _valid_teacher_lesson(target: TeacherTarget, task: str, lesson: str, result: str) -> bool:
    if any(_looks_repeated(value) for value in (task, lesson, result)):
        return False
    if target.dynamic_family != "two_steps":
        return True
    if _contains_bad_teacher_pattern(task, lesson, result):
        return False
    steps = _numbered_steps(result)
    if steps != ["1", "2"]:
        return False
    parts = re.split(r"(?:^|\s)[12][\.)]\s*", result)
    step_texts = [part.strip(" .;:") for part in parts if part.strip(" .;:")]
    if len(step_texts) != 2:
        return False
    if any(len(_plain_words(step)) < 3 for step in step_texts):
        return False
    return True


def _stable_id(*parts: str) -> str:
    raw = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _existing_teacher_signatures(store: MemoryStore) -> set[str]:
    signatures: set[str] = set()
    with store.connect() as conn:
        rows = conn.execute("SELECT context FROM experiences").fetchall()
    for (context,) in rows:
        if not isinstance(context, str) or not context.strip().startswith("{"):
            continue
        try:
            data = json.loads(context)
        except Exception:
            continue
        if data.get("source") == "ollama_teacher" and data.get("teacher_signature"):
            signatures.add(str(data["teacher_signature"]))
    return signatures


def _context(lesson: dict[str, str], cfg: dict[str, Any]) -> str:
    model = str(cfg.get("model") or DEFAULT_CONFIG["model"])
    return json.dumps(
        {
            "context": "Teacher-generated local practice packet. Accepted only as training material after Butterfly gates validate behavior.",
            "curriculum_node": lesson["curriculum_node"],
            "dynamic_family": lesson["dynamic_family"],
            "source": "ollama_teacher",
            "source_url": f"local://ollama/{model}",
            "source_title": f"Ollama teacher model {model}",
            "teacher_provider": "ollama",
            "teacher_model": model,
            "teacher_signature": lesson["signature"],
        },
        ensure_ascii=False,
    )


def _actions(lesson: dict[str, str], cfg: dict[str, Any]) -> list[dict[str, str]]:
    return [{
        "type": "teacher_learning_material",
        "curriculum_node": lesson["curriculum_node"],
        "dynamic_family": lesson["dynamic_family"],
        "source": "ollama_teacher",
        "teacher_model": str(cfg.get("model") or DEFAULT_CONFIG["model"]),
    }]


def install_teacher_lessons(
    *,
    store: MemoryStore | None = None,
    node: str | None = None,
    dynamic_family: str | None = None,
    count: int | None = None,
    dry_run: bool = False,
    cfg: dict[str, Any] | None = None,
    request_func: Callable[[str, dict[str, Any]], str] | None = None,
    mark_material_func: Callable[[str, str], None] | None = mark_material,
) -> dict[str, Any]:
    cfg = _deep_merge(load_teacher_config(), cfg or {})
    if not cfg.get("enabled", True):
        return {"enabled": False, "provider": cfg.get("provider"), "targets": [], "added": [], "skipped": []}
    if str(cfg.get("provider") or "ollama") != "ollama":
        raise RuntimeError("Only the free local Ollama teacher provider is currently enabled.")

    limits = cfg["limits"]
    lesson_count = int(count or limits["lessons_per_target"])
    targets = select_teacher_targets(
        node=node,
        dynamic_family=dynamic_family,
        limit=int(limits["targets_per_run"]),
    )
    if dry_run:
        return {
            "enabled": True,
            "provider": "ollama",
            "model": cfg.get("model"),
            "dry_run": True,
            "targets": [target.__dict__ for target in targets],
            "planned_lessons": lesson_count * len(targets),
            "added": [],
            "skipped": [],
        }

    store = store or MemoryStore()
    existing = _existing_teacher_signatures(store)
    request_func = request_func or _ollama_generate
    added: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for target in targets:
        lessons: list[dict[str, str]] = []
        lesson_signatures: set[str] = set()
        last_error: Exception | None = None
        for attempt in range(int(limits["retry_attempts"])):
            remaining = max(1, lesson_count - len(lessons))
            prompt = _prompt(target, remaining)
            if attempt:
                prompt += (
                    "\nTu respuesta anterior no alcanzo: fue JSON invalido, duplicado o insuficiente. "
                    f"Faltan {remaining} lecciones nuevas. Responde con JSON compacto y completo."
                )
            raw = request_func(prompt, cfg)
            try:
                parsed = parse_teacher_lessons(
                    raw,
                    target,
                    max_lesson_chars=int(limits["max_lesson_chars"]),
                )
            except (json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                continue
            for lesson in parsed:
                signature = lesson["signature"]
                if signature in lesson_signatures:
                    continue
                lesson_signatures.add(signature)
                lessons.append(lesson)
                if len(lessons) >= lesson_count:
                    break
            if len(lessons) >= lesson_count:
                break
        if not lessons:
            message = "Teacher did not return valid usable lessons."
            if last_error is not None:
                message += f" Last error: {type(last_error).__name__}: {last_error}"
            raise RuntimeError(message)
        for lesson in lessons[:lesson_count]:
            if lesson["signature"] in existing:
                skipped.append({"signature": lesson["signature"], "reason": "already_present"})
                continue
            if not dry_run:
                experience_id = store.add_experience(
                    task=lesson["task"],
                    result=lesson["result"],
                    lesson=lesson["lesson"],
                    context=_context(lesson, cfg),
                    actions=_actions(lesson, cfg),
                    verified=True,
                    quality=float(cfg.get("quality", 0.86)),
                )
                lesson = dict(lesson)
                lesson["experience_id"] = experience_id
                if mark_material_func is not None:
                    mark_material_func(lesson["curriculum_node"], "verified_packet")
            existing.add(lesson["signature"])
            added.append(dict(lesson))

    return {
        "enabled": True,
        "provider": "ollama",
        "model": cfg.get("model"),
        "dry_run": bool(dry_run),
        "targets": [target.__dict__ for target in targets],
        "added": added,
        "skipped": skipped,
    }


