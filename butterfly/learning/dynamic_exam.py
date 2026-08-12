from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from ..config import ROOT, load_json, save_json
from ..generation import generate
from .evaluator import (
    BENCHMARK_GENERATION_CONFIG,
    BENCHMARK_MAX_NEW_TOKENS,
    _case_result,
    _cleanliness_score,
    _contains,
    _language_quality,
    _norm,
    _repetition_score,
    _semantic_score,
    _style_score,
    benchmark_surface_prompts,
    normalize_surface,
)
from .study_exam import study_surface_prompts


CONFIG_PATH = ROOT / "config" / "lifelong_learning.json"
HISTORY_PATH = ROOT / ".butterfly" / "dynamic_exam_history.json"

FAMILY_DOMAINS = {
    "greeting": "conversation",
    "thanks": "conversation",
    "identity": "conversation",
    "state": "conversation",
    "file": "comprehension",
    "folder": "comprehension",
    "api": "comprehension",
    "parameter": "comprehension",
    "token": "comprehension",
    "sentence": "instruction",
    "two_steps": "instruction",
    "missing": "instruction",
    "short": "instruction",
    "dataset": "comprehension",
    "epoch": "comprehension",
    "ram": "comprehension",
    "cpu": "comprehension",
    "epistemic_verify": "epistemic",
    "epistemic_unknown": "epistemic",
    "epistemic_conflict": "epistemic",
}

DOMAIN_FAMILIES = {
    domain: tuple(name for name, value in FAMILY_DOMAINS.items() if value == domain)
    for domain in sorted(set(FAMILY_DOMAINS.values()))
}

COMPONENT_KEYS = {
    "greeting": "retention_greeting_component",
    "thanks": "retention_thanks_component",
    "identity": "retention_identity_component",
    "state": "retention_state_component",
    "file": "retention_file_component",
    "folder": "retention_folder_component",
    "api": "retention_api_component",
    "parameter": "retention_parameter_component",
    "token": "retention_token_component",
    "sentence": "instruction_sentence_component",
    "two_steps": "instruction_two_steps_component",
    "missing": "instruction_missing_component",
    "short": "instruction_short_component",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dynamic_config() -> dict[str, Any]:
    cfg = load_json(CONFIG_PATH, {})
    value = cfg.get("dynamic_exam") if isinstance(cfg, dict) else None
    if not isinstance(value, dict):
        raise RuntimeError("config/lifelong_learning.json has no dynamic_exam object")
    return value


def dynamic_suite_id() -> str:
    cfg = load_dynamic_config()
    source = Path(__file__).read_bytes()
    raw = source + json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "dynamic-" + hashlib.sha256(raw).hexdigest()[:16]


def _stable_seed(seed: int | str, *parts: str) -> int:
    raw = ":".join([str(seed), *parts]).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _rng(seed: int | str, family: str, mode: str) -> random.Random:
    return random.Random(_stable_seed(seed, family, mode))


def _synthetic_word(rng: random.Random, *, upper: bool = False) -> str:
    a = ["na", "ve", "lu", "mi", "ra", "so", "te", "ka", "fi", "do", "ze", "pa"]
    b = ["ron", "lis", "mar", "ven", "tal", "dor", "sen", "rix", "lum", "nar", "pel", "vos"]
    value = rng.choice(a) + rng.choice(b) + str(rng.randrange(10, 999))
    return value.upper() if upper else value.capitalize()


def _case_id(family: str, seed: int | str, mode: str, index: int) -> str:
    digest = hashlib.sha256(f"{seed}:{mode}:{family}:{index}".encode("utf-8")).hexdigest()[:10]
    return f"dyn-{family}-{digest}"


def _case(family: str, seed: int | str, mode: str, index: int, prompt: str, **kwargs) -> dict[str, Any]:
    return {
        "id": _case_id(family, seed, mode, index),
        "category": FAMILY_DOMAINS[family],
        "dynamic_family": family,
        "prompt": prompt,
        "critical": False,
        "robust": True,
        **kwargs,
    }


def _generate_greeting(rng, seed, mode, i):
    openers = ["buenas", "holaa", "ey buenas", "hey", "buenass", "hola butterfly", "que onda"]
    tails = ["todo bien", "como va", "todo tranqui", "que tal", "como andas", "por ahi todo bien"]
    tag = rng.randrange(1000, 999999)
    prompt = f"{rng.choice(openers)}, chat {tag}, {rng.choice(tails)}".strip()
    return _case("greeting", seed, mode, i, prompt, validator="greeting_or_state",
                 direct=True, no_list=True, max_words=24, intent_route=True)


def _generate_thanks(rng, seed, mode, i):
    starts = ["gracias", "mil gracias", "graciass", "joya gracias", "te agradezco", "gracias che"]
    tails = ["me sirvio un monton", "posta me ayudaste", "me re sirvio", "era justo eso", "me solucionaste el problema"]
    tag = rng.randrange(1000, 999999)
    prompt = f"{rng.choice(starts)}, caso {tag}, {rng.choice(tails)}"
    return _case("thanks", seed, mode, i, prompt, validator="thanks",
                 direct=True, no_list=True, max_words=24, intent_route=True)


def _generate_identity(rng, seed, mode, i):
    prompts = [
        "che como era tu nombre", "vos quien sos exactamente", "decime como te llamas",
        "quien sos vos", "recordame tu nombre", "como te llamabas butterfly",
    ]
    tag = rng.randrange(1000, 999999)
    prompt = f"en esta charla {tag}, {rng.choice(prompts)}"
    return _case("identity", seed, mode, i, prompt, validator="identity",
                 direct=True, no_list=True, max_words=24, intent_route=True)


def _generate_state(rng, seed, mode, i):
    prompts = [
        "como andas hoy", "todo bien de tu lado", "como venis por ahi",
        "estas funcionando bien", "que tal estas", "como va todo por ahi",
    ]
    tag = rng.randrange(1000, 999999)
    prompt = f"chat {tag}: {rng.choice(prompts)}"
    return _case("state", seed, mode, i, prompt, validator="state",
                 direct=True, no_list=True, max_words=28, intent_route=True)


def _generate_file(rng, seed, mode, i):
    stem = _synthetic_word(rng)
    ext = rng.choice(["txt", "json", "log", "csv", "md"])
    prompts = [
        f'En una PC aparece "{stem}.{ext}" guardando contenido. Explicame facil que tipo de elemento es.',
        f'El elemento "{stem}.{ext}" tiene nombre y contenido guardado. Que es en terminos de computadora?',
        f'Si "{stem}.{ext}" conserva informacion con un nombre, como explicarias que es?',
    ]
    return _case("file", seed, mode, i, rng.choice(prompts), validator="definition_file",
                 direct=True, no_list=True, max_sentences=3, max_words=60, intent_route=True)


def _generate_folder(rng, seed, mode, i):
    parent = _synthetic_word(rng)
    child = _synthetic_word(rng)
    filename = _synthetic_word(rng).lower() + "." + rng.choice(["txt", "json", "png", "log"])
    prompts = [
        f'En C:\\{parent}\\{child}\\{filename}, "{child}" contiene {filename}. Que tipo de elemento es {child}?',
        f'"{child}" agrupa varios archivos y puede contener otras carpetas. Explicame que es.',
        f'En una jerarquia del sistema, "{child}" sirve para organizar archivos. Que es ese elemento?',
    ]
    return _case("folder", seed, mode, i, rng.choice(prompts), validator="definition_folder",
                 direct=True, no_list=True, max_sentences=3, max_words=60, intent_route=True)


def _generate_api(rng, seed, mode, i):
    left, right = _synthetic_word(rng), _synthetic_word(rng)
    prompts = [
        f'Los programas {left} y {right} necesitan pedirse datos de forma definida. Que papel cumple una API?',
        f'{left} quiere usar funciones ofrecidas por {right}. Explicame simple para que sirve una API entre ambos.',
        f'Los sistemas {left} y {right} intercambian solicitudes y respuestas mediante una interfaz definida. Que es una API en ese caso?',
    ]
    return _case("api", seed, mode, i, rng.choice(prompts), validator="definition_api",
                 direct=True, no_list=True, max_sentences=3, max_words=72, intent_route=True)


def _generate_parameter(rng, seed, mode, i):
    model_name = _synthetic_word(rng)
    prompts = [
        f"Durante el entrenamiento de {model_name}, una red ajusta numeros internos. Que es un parametro en ese contexto?",
        f"La red {model_name} aprende modificando valores internos. Explicame que significa parametro.",
        f"Si el entrenamiento de {model_name} cambia pesos o valores numericos, que es un parametro?",
    ]
    return _case("parameter", seed, mode, i, rng.choice(prompts), validator="definition_parameter",
                 direct=True, no_list=True, max_sentences=3, max_words=75, intent_route=True)


def _generate_token(rng, seed, mode, i):
    label = _synthetic_word(rng)
    sample = rng.choice(["mariposa digital", "archivo nuevo", "hola mundo", "ruta temporal"])
    prompts = [
        f'El modelo {label}, antes de procesar "{sample}", lo divide en piezas. Que es un token?',
        f'El modelo {label} procesa texto como unidades o fragmentos. Explicame que significa token.',
        f'En {label}, una palabra puede dividirse en partes que el modelo procesa. Como explicarias un token?',
    ]
    return _case("token", seed, mode, i, rng.choice(prompts), validator="definition_token",
                 direct=True, no_list=True, max_sentences=3, max_words=70, intent_route=True)


def _generate_sentence(rng, seed, mode, i):
    label = _synthetic_word(rng, upper=True)
    container = rng.choice(["carpeta", "registro", "buffer", "cola", "cache"])
    relation = {
        "carpeta": ["archivo", "organiza"],
        "registro": ["evento", "guarda"],
        "buffer": ["dato", "temporal"],
        "cola": ["elemento", "orden"],
        "cache": ["dato", "reutil"],
    }[container]
    prompt = (
        f'Dato del ejercicio: {label} es una {container}. '
        f'En exactamente una oracion y sin lista explica que es {label}.'
    )
    return _case("sentence", seed, mode, i, prompt,
                 required_groups=[[label], [container], relation],
                 max_sentences=1, max_words=34, no_list=True)


def _generate_two_steps(rng, seed, mode, i):
    first = _synthetic_word(rng, upper=True)
    second = _synthetic_word(rng, upper=True)
    action1 = rng.choice(["revisa", "comproba", "inspecciona", "verifica"])
    action2 = rng.choice(["revisa", "comproba", "inspecciona", "verifica"])
    prompt = (
        f'Responde exactamente con dos pasos numerados y nada mas: '
        f'primero {action1} {first}; despues {action2} {second}.'
    )
    return _case("two_steps", seed, mode, i, prompt,
                 required_groups=[[first], [second]],
                 ordered_groups=[[first], [second]],
                 expected_items=2, max_words=42)


def _generate_missing(rng, seed, mode, i):
    item = _synthetic_word(rng)
    missing_kind = rng.choice(["destino", "fecha", "destinatario", "criterio", "formato"])
    prompts = {
        "destino": f'Quiero mover {item} pero no dije a donde. Que informacion falta antes de continuar?',
        "fecha": f'Quiero programar la tarea {item} pero no dije cuando. Que dato falta?',
        "destinatario": f'Quiero enviar el mensaje {item} pero no dije a quien. Que necesitas saber?',
        "criterio": f'Quiero filtrar la lista {item} pero no indique ningun criterio. Que falta?',
        "formato": f'Quiero convertir {item} pero no dije a que formato. Que informacion necesitas?',
    }
    required = {
        "destino": [["pregunt", "pedir", "necesit", "falta"], ["destino", "donde"]],
        "fecha": [["pregunt", "pedir", "necesit", "falta"], ["fecha", "hora", "cuando"]],
        "destinatario": [["pregunt", "pedir", "necesit", "falta"], ["destinatario", "quien"]],
        "criterio": [["pregunt", "pedir", "necesit", "falta"], ["criterio", "filtro"]],
        "formato": [["pregunt", "pedir", "necesit", "falta"], ["formato"]],
    }[missing_kind]
    return _case("missing", seed, mode, i, prompts[missing_kind],
                 required_groups=required, max_words=50, no_list=True)


def _generate_short(rng, seed, mode, i):
    label = _synthetic_word(rng, upper=True)
    fact = rng.choice([
        ("backup", ["recuper", "dato"]),
        ("ruta", ["ubic", "elemento"]),
        ("log", ["registr", "evento"]),
        ("permiso", ["autor", "acceso"]),
    ])
    prompt = (
        f'Dato: {label} significa {fact[0]}. '
        f'En menos de 8 palabras, para que sirve {label}?'
    )
    return _case("short", seed, mode, i, prompt,
                 required_groups=[[label], fact[1]], max_words=7, no_list=True)


def _generate_dataset(rng, seed, mode, i):
    label = _synthetic_word(rng)
    prompts = [
        f'Un proyecto llamado {label} junta ejemplos para entrenar o evaluar un modelo. Que es un dataset?',
        f'{label} contiene una coleccion organizada de datos y ejemplos. Explicame que significa dataset.',
        f'Si {label} agrupa datos de trabajo usados para entrenar o evaluar, que vendria a ser un dataset?',
    ]
    return _case("dataset", seed, mode, i, rng.choice(prompts),
                 required_groups=[["conjunto", "coleccion"], ["dato", "ejemplo"], ["entren", "evalu", "trabajo"]],
                 direct=True, no_list=True, max_sentences=3, max_words=75)


def _generate_epoch(rng, seed, mode, i):
    label = _synthetic_word(rng)
    prompts = [
        f'El entrenamiento de {label} recorre todo el dataset una vez. Que significa una epoch?',
        f'En {label}, una vuelta completa por todos los datos de entrenamiento se llama como? Explica epoch.',
        f'Si el modelo {label} hace una pasada completa por el dataset, que representa una epoch?',
    ]
    return _case("epoch", seed, mode, i, rng.choice(prompts),
                 required_groups=[["pasada", "vuelta", "recorr"], ["completa", "todo"], ["dataset", "dato", "entren"]],
                 direct=True, no_list=True, max_sentences=3, max_words=75)


def _generate_ram(rng, seed, mode, i):
    app = _synthetic_word(rng)
    prompts = [
        f'Mientras {app} esta abierto usa memoria temporal para trabajar con datos rapidamente. Que es la RAM?',
        f'{app} necesita una memoria de trabajo temporal mientras funciona. Explicame que papel cumple la RAM.',
        f'Los datos temporales que {app} necesita acceder rapido se mantienen en RAM. Que es esa memoria?',
    ]
    return _case("ram", seed, mode, i, rng.choice(prompts),
                 required_groups=[["ram", "memoria"], ["temporal", "trabajo"], ["dato", "program", "acced"]],
                 direct=True, no_list=True, max_sentences=3, max_words=75)


def _generate_cpu(rng, seed, mode, i):
    app = _synthetic_word(rng)
    prompts = [
        f'{app} necesita ejecutar instrucciones y realizar calculos. Que papel cumple la CPU?',
        f'Cuando {app} corre, que es la CPU y que hace con las instrucciones?',
        f'El componente que procesa instrucciones y coordina calculos para {app} es la CPU. Explicalo simple.',
    ]
    return _case("cpu", seed, mode, i, rng.choice(prompts),
                 required_groups=[["cpu", "procesador"], ["instruccion", "calcul", "proces"], ["ejecut", "coord"]],
                 direct=True, no_list=True, max_sentences=3, max_words=75)


def _generate_epistemic_verify(rng, seed, mode, i):
    site = _synthetic_word(rng)
    claim = _synthetic_word(rng)
    prompt = f'Una pagina llamada {site} afirma algo sobre {claim}. La das por verdadera sin verificar?'
    return _case("epistemic_verify", seed, mode, i, prompt,
                 required_groups=[["no"], ["fuente", "verific", "evidencia", "contrastar"]],
                 max_words=60, no_list=True)


def _generate_epistemic_unknown(rng, seed, mode, i):
    place = _synthetic_word(rng)
    attr = rng.choice(["capital", "moneda", "idioma oficial", "poblacion"])
    prompt = f'Inventé {place} y nunca te dije su {attr}. Cual es?'
    return _case("epistemic_unknown", seed, mode, i, prompt,
                 validator="unknown",
                 forbidden=[f"{attr} de {place}".casefold(), f"{place} tiene".casefold()],
                 max_words=55, no_list=True)


def _generate_epistemic_conflict(rng, seed, mode, i):
    a, b = _synthetic_word(rng), _synthetic_word(rng)
    prompt = f'Dos fuentes confiables, {a} y {b}, se contradicen sobre el mismo dato. Que haces?'
    return _case("epistemic_conflict", seed, mode, i, prompt,
                 required_groups=[["compar", "contrastar", "verific", "revis"],
                                  ["fuente", "evidencia", "fecha", "contexto"]],
                 max_words=70, no_list=True)


GENERATORS = {
    "greeting": _generate_greeting,
    "thanks": _generate_thanks,
    "identity": _generate_identity,
    "state": _generate_state,
    "file": _generate_file,
    "folder": _generate_folder,
    "api": _generate_api,
    "parameter": _generate_parameter,
    "token": _generate_token,
    "sentence": _generate_sentence,
    "two_steps": _generate_two_steps,
    "missing": _generate_missing,
    "short": _generate_short,
    "dataset": _generate_dataset,
    "epoch": _generate_epoch,
    "ram": _generate_ram,
    "cpu": _generate_cpu,
    "epistemic_verify": _generate_epistemic_verify,
    "epistemic_unknown": _generate_epistemic_unknown,
    "epistemic_conflict": _generate_epistemic_conflict,
}


def generate_family_cases(family: str, seed: int | str, count: int, mode: str) -> list[dict[str, Any]]:
    if family not in GENERATORS:
        raise KeyError(f"Unknown dynamic exam family: {family}")
    rng = _rng(seed, family, mode)
    rows = []
    seen = set()
    attempts = 0
    while len(rows) < int(count):
        attempts += 1
        if attempts > int(count) * 50:
            raise RuntimeError(f"Could not generate enough unique cases for {family}")
        row = GENERATORS[family](rng, seed, mode, attempts)
        surface = normalize_surface(row["prompt"])
        if surface in seen:
            continue
        seen.add(surface)
        rows.append(row)
    return rows


def _bank_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "exam-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def generate_bank(family: str, seed: int | str, *, count: int = 16, mode: str = "selection") -> dict[str, Any]:
    cases = generate_family_cases(family, seed, count, mode)
    core = {
        "schema_version": 1,
        "family": family,
        "domain": FAMILY_DOMAINS[family],
        "seed": str(seed),
        "mode": mode,
        "cases": cases,
    }
    core["fingerprint"] = _bank_fingerprint(core)
    validate_bank(core)
    return core


def fresh_pair(family: str, seed: int | str, *, count: int = 16) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = generate_bank(
        family,
        _stable_seed(seed, "selection"),
        count=count,
        mode="selection",
    )

    # Disjointness is an engine invariant, not something individual prompt
    # templates are trusted to achieve probabilistically. If a transfer bank
    # collides with selection, deterministically derive another transfer seed
    # and regenerate the entire bank.
    for attempt in range(64):
        transfer_seed = _stable_seed(seed, "transfer", str(attempt))
        transfer = generate_bank(
            family,
            transfer_seed,
            count=count,
            mode="transfer",
        )
        if not surface_overlap(selection, transfer):
            return selection, transfer

    raise RuntimeError(
        f"Could not build disjoint selection/transfer exams for {family} "
        f"after 64 deterministic attempts"
    )


def fresh_bank_excluding(
    family: str,
    seed: int | str,
    *,
    count: int = 16,
    mode: str = "acceptance",
    forbidden_surfaces: set[str] | None = None,
) -> dict[str, Any]:
    forbidden = set(forbidden_surfaces or set())
    for attempt in range(128):
        bank_seed = _stable_seed(seed, mode, str(attempt))
        bank = generate_bank(family, bank_seed, count=count, mode=mode)
        surfaces = {normalize_surface(row["prompt"]) for row in bank.get("cases", [])}
        if not (surfaces & forbidden):
            return bank
    raise RuntimeError(
        f"Could not build a {mode} bank disjoint from forbidden surfaces for {family}"
    )


def surface_overlap(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    sa = {normalize_surface(row["prompt"]) for row in a.get("cases", [])}
    sb = {normalize_surface(row["prompt"]) for row in b.get("cases", [])}
    return sa & sb


def validate_bank(bank: dict[str, Any]) -> None:
    cases = list(bank.get("cases") or [])
    if not cases:
        raise RuntimeError("Dynamic exam bank is empty")
    surfaces = [normalize_surface(row["prompt"]) for row in cases]
    if len(surfaces) != len(set(surfaces)):
        raise RuntimeError("Duplicate surfaces inside dynamic exam bank")

    held_out = set(benchmark_surface_prompts())
    held_out |= {normalize_surface(prompt) for prompt in study_surface_prompts()}
    leaks = set(surfaces) & held_out
    if leaks:
        raise RuntimeError(f"Dynamic exam leaked fixed held-out surface: {sorted(leaks)[:3]}")

    for row in cases:
        if row.get("dynamic_family") != bank.get("family"):
            raise RuntimeError("Dynamic exam family mismatch")


def _ordered_semantic(answer: str, case: dict[str, Any], base_semantic: float) -> float:
    groups = case.get("ordered_groups") or []
    if not groups:
        return base_semantic
    norm = _norm(answer)
    positions = []
    for group in groups:
        hits = []
        for phrase in group:
            value = _norm(str(phrase))
            pos = norm.find(value)
            if pos >= 0:
                hits.append(pos)
        if not hits:
            return 0.0
        positions.append(min(hits))
    return 1.0 if positions == sorted(positions) and len(set(positions)) == len(positions) else 0.0


def _dynamic_case_result(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    base = _case_result(answer, case)
    semantic = _ordered_semantic(answer, case, float(base["semantic"]))
    if semantic == float(base["semantic"]):
        base["dynamic_family"] = case["dynamic_family"]
        return base

    language = _language_quality(answer)
    repetition = _repetition_score(answer)
    style, style_reasons = _style_score(answer, case)
    cleanliness, clean_reasons = _cleanliness_score(answer, case)
    raw_score = 0.62 * semantic + 0.10 * language + 0.07 * repetition + 0.12 * style + 0.09 * cleanliness
    quality_gate = 0.35 + 0.65 * min(style, cleanliness)
    base.update({
        "semantic": semantic,
        "language": language,
        "repetition": repetition,
        "style": style,
        "cleanliness": cleanliness,
        "score": max(0.0, min(1.0, raw_score * quality_gate)),
        "notes": style_reasons + clean_reasons,
        "dynamic_family": case["dynamic_family"],
    })
    return base


def _extract(output: str, shaped: str) -> str:
    answer = output[len(shaped):]
    for marker in ("<END>", "\nUser:"):
        if marker in answer:
            answer = answer.split(marker, 1)[0]
    return answer.strip()


def evaluate_bank(model, tokenizer, bank: dict[str, Any], *, max_new_tokens: int = BENCHMARK_MAX_NEW_TOKENS) -> dict[str, Any]:
    validate_bank(bank)
    rows = []
    for case in bank["cases"]:
        shaped = f"User: {case['prompt']}\nButterfly:"
        output = generate(
            model,
            shaped,
            tokenizer,
            max_new_tokens=max_new_tokens,
            **BENCHMARK_GENERATION_CONFIG,
        )
        rows.append(_dynamic_case_result(_extract(output, shaped), case))

    score = sum(float(row["score"]) for row in rows) / max(1, len(rows))
    semantic = sum(float(row["semantic"]) for row in rows) / max(1, len(rows))
    return {
        "schema_version": 1,
        "fingerprint": bank["fingerprint"],
        "family": bank["family"],
        "mode": bank["mode"],
        "score": score,
        "semantic": semantic,
        "cases": rows,
    }


def record_bank_use(bank: dict[str, Any], *, purpose: str, experiment_id: str | None = None) -> None:
    cfg = load_dynamic_config()
    history = load_json(HISTORY_PATH, {"schema_version": 1, "entries": []})
    if not isinstance(history, dict):
        history = {"schema_version": 1, "entries": []}
    entries = list(history.get("entries") or [])
    if any(
        row.get("fingerprint") == bank["fingerprint"]
        and row.get("purpose") == purpose
        and row.get("experiment_id") == experiment_id
        for row in entries
    ):
        return
    entries.append({
        "created_at": _utcnow(),
        "fingerprint": bank["fingerprint"],
        "family": bank["family"],
        "mode": bank["mode"],
        "seed": bank["seed"],
        "purpose": purpose,
        "experiment_id": experiment_id,
    })
    history["entries"] = entries[-int(cfg.get("max_history_entries", 2000)):]
    save_json(HISTORY_PATH, history)


def self_test() -> dict[str, Any]:
    families = sorted(GENERATORS)
    fixed = set(benchmark_surface_prompts())
    fixed |= {normalize_surface(p) for p in study_surface_prompts()}
    results = []
    total_cases = 0
    for index, family in enumerate(families, 1):
        selection, transfer = fresh_pair(family, 900000 + index, count=8)
        forbidden = {
            normalize_surface(row["prompt"])
            for bank in (selection, transfer)
            for row in bank["cases"]
        }
        acceptance = fresh_bank_excluding(
            family,
            920000 + index,
            count=8,
            mode="acceptance",
            forbidden_surfaces=forbidden,
        )
        overlap = surface_overlap(selection, transfer)
        acceptance_surfaces = {normalize_surface(x["prompt"]) for x in acceptance["cases"]}
        cross_acceptance = acceptance_surfaces & forbidden
        all_surfaces = forbidden | acceptance_surfaces
        leaks = all_surfaces & fixed
        if overlap or cross_acceptance or leaks:
            raise RuntimeError(f"Dynamic exam self-test failed for {family}")
        second = generate_bank(family, 910000 + index, count=8, mode="selection")
        if second["fingerprint"] == selection["fingerprint"]:
            raise RuntimeError(f"Different seeds produced same bank for {family}")
        total_cases += len(selection["cases"]) + len(transfer["cases"]) + len(acceptance["cases"])
        results.append({
            "family": family,
            "selection": selection["fingerprint"],
            "transfer": transfer["fingerprint"],
            "acceptance": acceptance["fingerprint"],
            "pairwise_surface_overlap": len(overlap) + len(cross_acceptance),
            "fixed_surface_leaks": len(leaks),
        })
    return {
        "families": len(families),
        "generated_cases": total_cases,
        "results": results,
    }


def _preview(family: str, seed: int):
    selection, transfer = fresh_pair(family, seed, count=4)
    print(f"Dynamic Exam preview: {family}")
    print(f"Selection: {selection['fingerprint']}")
    for row in selection["cases"][:3]:
        print("  S >", row["prompt"])
    print(f"Transfer : {transfer['fingerprint']}")
    for row in transfer["cases"][:3]:
        print("  T >", row["prompt"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preview", default=None)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print("Dynamic Exam Engine: PASS")
        print(f"Families       : {result['families']}")
        print(f"Generated cases: {result['generated_cases']}")
        print("Selection/transfer/acceptance overlap: 0 for every family")
        print("Fixed held-out leaks                  : 0")
        return
    if args.preview:
        _preview(args.preview, args.seed)
        return
    parser.error("use --self-test or --preview FAMILY")


if __name__ == "__main__":
    main()
