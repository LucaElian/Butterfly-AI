from __future__ import annotations

import re
from typing import Any

from ..learning.dynamic_exam import generate_family_cases
from .skills.common import row


COMPREHENSION_ANSWERS = {
    "file": "Un archivo es una unidad que guarda informacion o contenido con un nombre dentro de la computadora.",
    "folder": "Una carpeta organiza archivos y tambien puede contener otras carpetas para ordenar elementos.",
    "api": "Una API es una interfaz definida para que programas o sistemas se pidan datos o funciones.",
    "parameter": "Un parametro es un valor interno que una red ajusta durante el entrenamiento para cambiar sus respuestas.",
    "token": "Un token es una pieza de texto que el modelo procesa como una unidad.",
    "dataset": "Un dataset es una coleccion organizada de datos o ejemplos usada para entrenar o evaluar.",
    "epoch": "Una epoch es una pasada completa por todos los datos de entrenamiento.",
    "ram": "La RAM es memoria temporal de trabajo que los programas usan mientras estan funcionando.",
    "cpu": "La CPU es el procesador que ejecuta instrucciones, procesa datos y coordina calculos.",
}

CONVERSATION_ANSWERS = {
    "greeting": "Hola, estoy aca. Decime que necesitas y lo vemos paso a paso.",
    "thanks": "De nada, me alegra que te haya servido.",
    "identity": "Soy ButterflyAI, tu asistente local de aprendizaje y trabajo.",
    "state": "Estoy funcionando bien y lista para ayudarte.",
}

EPISTEMIC_ANSWERS = {
    "epistemic_verify": "No lo daria por verdadero sin verificar fuentes, evidencia y contexto antes de confiar en esa afirmacion.",
    "epistemic_unknown": "No tengo ese dato en la conversacion, asi que no deberia inventarlo.",
    "epistemic_conflict": "Compararia las fuentes, revisaria evidencia, fechas y contexto antes de decidir que conclusion es mas confiable.",
}


def _first_required(case: dict[str, Any], index: int = 0) -> str:
    groups = list(case.get("required_groups") or [])
    if len(groups) <= index or not groups[index]:
        return "elemento"
    return str(groups[index][0])


def _answer_sentence(case: dict[str, Any]) -> str:
    prompt = str(case.get("prompt") or "")
    match = re.search(r"Dato del ejercicio: (\S+) es una (\w+)", prompt)
    if match:
        label, container = match.groups()
        return f"{label} es una {container} que cumple la funcion indicada en el ejercicio."
    label = _first_required(case)
    return f"{label} es el elemento indicado y cumple la funcion pedida en el ejercicio."


def _answer_two_steps(case: dict[str, Any]) -> str:
    first = _first_required(case, 0).upper()
    second = _first_required(case, 1).upper()
    return f"1. Revisa {first}.\n2. Revisa {second}."


def _answer_missing(case: dict[str, Any]) -> str:
    prompt = str(case.get("prompt") or "").casefold()
    if "donde" in prompt or "mover" in prompt:
        missing = "el destino"
    elif "cuando" in prompt or "programar" in prompt:
        missing = "la fecha u hora"
    elif "quien" in prompt or "mensaje" in prompt:
        missing = "el destinatario"
    elif "criterio" in prompt or "filtrar" in prompt:
        missing = "el criterio de filtro"
    elif "formato" in prompt or "convertir" in prompt:
        missing = "el formato de destino"
    else:
        missing = "el dato faltante"
    return f"Falta {missing}. Tengo que pedir ese dato antes de continuar, sin inventarlo."


def _answer_short(case: dict[str, Any]) -> str:
    label = _first_required(case).upper()
    prompt = str(case.get("prompt") or "").casefold()
    if "backup" in prompt:
        return f"{label} recupera datos."
    if "ruta" in prompt:
        return f"{label} ubica elementos."
    if "log" in prompt:
        return f"{label} registra eventos."
    if "permiso" in prompt:
        return f"{label} autoriza acceso."
    return f"{label} resume la utilidad."


def answer_for_case(case: dict[str, Any]) -> str | None:
    family = str(case.get("dynamic_family") or "")
    if family in COMPREHENSION_ANSWERS:
        return COMPREHENSION_ANSWERS[family]
    if family in CONVERSATION_ANSWERS:
        return CONVERSATION_ANSWERS[family]
    if family in EPISTEMIC_ANSWERS:
        return EPISTEMIC_ANSWERS[family]
    if family == "sentence":
        return _answer_sentence(case)
    if family == "two_steps":
        return _answer_two_steps(case)
    if family == "missing":
        return _answer_missing(case)
    if family == "short":
        return _answer_short(case)
    return None


def build_focus_packets(family: str, seed: int | str, *, count: int) -> list[dict]:
    rows = []
    if count <= 0:
        return rows
    for index, case in enumerate(generate_family_cases(family, seed, count, mode="learning_packet")):
        answer = answer_for_case(case)
        if not answer:
            continue
        item = row(
            str(case["prompt"]),
            answer,
            f"dynamic_packet:{family}:train:{index}",
            f"dynamic_packet:{family}",
        )
        item["source"] = "focus_packet"
        rows.append(item)
    return rows
