from __future__ import annotations

import json
from typing import Any, Callable, Iterable

from ..memory import MemoryStore
from .curriculum_graph import mark_material


CURATED_RESEARCH_SEED: list[dict[str, Any]] = [
    {
        "curriculum_node": "programming.api_basics",
        "dynamic_family": "api",
        "source_title": "MDN Web Docs - Web APIs",
        "source_url": "https://developer.mozilla.org/en-US/docs/Web/API",
        "task": "Estudiar que es una API en programacion.",
        "lesson": "Una API define una interfaz documentada para que programas o sistemas pidan datos, capacidades o funciones sin depender de los detalles internos de implementacion.",
    },
    {
        "curriculum_node": "programming.api_basics",
        "dynamic_family": "api",
        "source_title": "MDN Web Docs - Glossary",
        "source_url": "https://developer.mozilla.org/en-US/docs/Glossary",
        "task": "Diferenciar una API de la implementacion interna.",
        "lesson": "La API describe como interactuar con una capacidad; la implementacion interna es el codigo o sistema que hace el trabajo por debajo.",
    },
    {
        "curriculum_node": "computing.files",
        "dynamic_family": "file",
        "source_title": "Microsoft Support - File Explorer in Windows",
        "source_url": "https://support.microsoft.com/en-us/windows/experience/fileexplorer/file-explorer-in-windows",
        "task": "Estudiar que hace un archivo en Windows.",
        "lesson": "Un archivo guarda contenido o datos con un nombre; File Explorer permite encontrar, abrir, mover, compartir y administrar archivos en Windows.",
    },
    {
        "curriculum_node": "computing.folders",
        "dynamic_family": "folder",
        "source_title": "Microsoft Support - File Explorer in Windows",
        "source_url": "https://support.microsoft.com/en-us/windows/experience/fileexplorer/file-explorer-in-windows",
        "task": "Estudiar para que sirve una carpeta.",
        "lesson": "Una carpeta sirve para organizar archivos y otras carpetas; en File Explorer se usa para ordenar, encontrar y mover elementos dentro del equipo.",
    },
    {
        "curriculum_node": "computing.cpu",
        "dynamic_family": "cpu",
        "source_title": "Microsoft Support - All about processors CPUs",
        "source_url": "https://support.microsoft.com/es-ES/Windows/experience/compatibility/all-about-processors-cpus",
        "task": "Estudiar que funcion cumple la CPU.",
        "lesson": "La CPU es la unidad central de procesamiento: ejecuta instrucciones, coordina componentes y procesa tareas del sistema y los programas.",
    },
    {
        "curriculum_node": "computing.memory",
        "dynamic_family": "ram",
        "source_title": "IBM History - Dynamic random-access memory",
        "source_url": "https://www.ibm.com/history/dram",
        "task": "Estudiar que es la memoria RAM o DRAM.",
        "lesson": "La RAM es memoria de trabajo temporal usada por la computadora para mantener datos que necesita procesar rapidamente mientras ejecuta programas.",
    },
    {
        "curriculum_node": "ai.tokens",
        "dynamic_family": "token",
        "source_title": "IBM watsonx - Tokens and tokenization",
        "source_url": "https://www.ibm.com/docs/en/watsonx/saas?topic=solutions-tokens",
        "task": "Estudiar que es un token en modelos de lenguaje.",
        "lesson": "Un token es una unidad de texto que el modelo procesa; la tokenizacion convierte el prompt y la salida en unidades que pueden representar palabras, partes de palabras, espacios o signos.",
    },
    {
        "curriculum_node": "ai.parameters",
        "dynamic_family": "parameter",
        "source_title": "Python Documentation - Glossary",
        "source_url": "https://docs.python.org/3/glossary.html",
        "task": "Estudiar la idea general de parametro usando documentacion tecnica.",
        "lesson": "Un parametro es un nombre definido por una funcion o sistema para recibir un valor; en aprendizaje automatico, un parametro tambien puede ser un valor interno ajustado durante entrenamiento.",
    },
    {
        "curriculum_node": "instruction.two_steps",
        "dynamic_family": "two_steps",
        "source_title": "Microsoft Style Guide - Writing step-by-step instructions",
        "source_url": "https://learn.microsoft.com/en-us/style-guide/procedures-instructions/writing-step-by-step-instructions",
        "task": "Estudiar como responder instrucciones de varios pasos.",
        "lesson": "Cuando una tarea tiene pasos ordenados, conviene usar una lista numerada, separar una accion por paso y empezar con verbos claros e imperativos.",
    },
    {
        "curriculum_node": "instruction.sentence",
        "dynamic_family": "sentence",
        "source_title": "Google Technical Writing - Lists and tables",
        "source_url": "https://developers.google.com/tech-writing/one/lists-and-tables",
        "task": "Estudiar claridad y paralelismo en respuestas tecnicas.",
        "lesson": "Las respuestas tecnicas deben organizar informacion con estructura clara; las listas y frases paralelas ayudan a que el lector entienda y compare elementos.",
    },
    {
        "curriculum_node": "instruction.short",
        "dynamic_family": "short",
        "source_title": "Google Developer Documentation Style Guide",
        "source_url": "https://developers.google.com/style",
        "task": "Estudiar respuestas cortas y claras.",
        "lesson": "Una respuesta corta debe priorizar claridad, consistencia y utilidad para el lector; conviene evitar relleno y conservar el dato esencial.",
    },
    {
        "curriculum_node": "instruction.missing",
        "dynamic_family": "missing",
        "source_title": "Microsoft Style Guide - Procedures and instructions",
        "source_url": "https://learn.microsoft.com/en-us/style-guide/procedures-instructions/",
        "task": "Estudiar que hacer cuando falta informacion para completar una tarea.",
        "lesson": "Si una instruccion no tiene los datos necesarios para completarse de forma segura, hay que pedir el dato faltante antes de inventar o ejecutar una accion ambigua.",
    },
]


def _context(entry: dict[str, Any]) -> str:
    return json.dumps(
        {
            "context": "Curated research seed installed locally from a trusted source reference.",
            "curriculum_node": entry["curriculum_node"],
            "dynamic_family": entry["dynamic_family"],
            "source": "curated_research_seed",
            "source_url": entry["source_url"],
            "source_title": entry["source_title"],
        },
        ensure_ascii=False,
    )


def _actions(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "type": "curated_research_seed",
        "curriculum_node": entry["curriculum_node"],
        "dynamic_family": entry["dynamic_family"],
        "source_url": entry["source_url"],
        "source_title": entry["source_title"],
    }]


def _existing_keys(store: MemoryStore) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    with store.connect() as conn:
        rows = conn.execute("SELECT context FROM experiences").fetchall()
    for (context,) in rows:
        if not isinstance(context, str) or not context.strip().startswith("{"):
            continue
        try:
            data = json.loads(context)
        except Exception:
            continue
        node = data.get("curriculum_node")
        source_url = data.get("source_url")
        if node and source_url:
            keys.add((str(node), str(source_url)))
    return keys


def install_curated_research_seed(
    *,
    store: MemoryStore | None = None,
    nodes: Iterable[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    mark_material_func: Callable[[str, str], None] | None = mark_material,
) -> dict[str, Any]:
    store = store or MemoryStore()
    selected_nodes = {str(node) for node in nodes or []}
    existing = _existing_keys(store)
    added: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    candidates = [
        entry for entry in CURATED_RESEARCH_SEED
        if not selected_nodes or str(entry["curriculum_node"]) in selected_nodes
    ]
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]

    for entry in candidates:
        key = (str(entry["curriculum_node"]), str(entry["source_url"]))
        if key in existing:
            skipped.append({"curriculum_node": key[0], "source_url": key[1], "reason": "already_present"})
            if not dry_run and mark_material_func is not None:
                mark_material_func(str(entry["curriculum_node"]), "verified_packet")
            continue
        if not dry_run:
            experience_id = store.add_experience(
                task=entry["task"],
                result="Curated verified research seed installed.",
                lesson=entry["lesson"],
                context=_context(entry),
                actions=_actions(entry),
                verified=True,
                quality=0.95,
            )
            if mark_material_func is not None:
                mark_material_func(str(entry["curriculum_node"]), "verified_packet")
            entry = dict(entry)
            entry["experience_id"] = experience_id
        added.append(dict(entry))
        existing.add(key)

    return {
        "available": len(CURATED_RESEARCH_SEED),
        "selected": len(candidates),
        "added": added,
        "skipped": skipped,
        "dry_run": bool(dry_run),
    }