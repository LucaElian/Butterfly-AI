from __future__ import annotations

from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import re

from ..config import ROOT, CORPUS_DIR, save_json
from ..learning.evaluator import CASES

V5_DIR = CORPUS_DIR / "v0005"
MANIFEST = V5_DIR / "manifest.json"

STAGE_FILES = {
    "basic_dialogue": (V5_DIR / "basic_train.jsonl", V5_DIR / "basic_valid.jsonl"),
    "instruction_following": (V5_DIR / "instruction_train.jsonl", V5_DIR / "instruction_valid.jsonl"),
    "epistemic_dialogue": (V5_DIR / "epistemic_train.jsonl", V5_DIR / "epistemic_valid.jsonl"),
    "mixed_consolidation": (V5_DIR / "mixed_train.jsonl", V5_DIR / "mixed_valid.jsonl"),
}


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .!?¿¡")


BENCHMARK_PROMPTS = {_norm(c["prompt"]) for c in CASES}


def _row(user: str, assistant: str, category: str) -> dict:
    return {"user": user.strip(), "assistant": assistant.strip(), "category": category}


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = _norm(row["user"])
        if not key or key in seen:
            continue
        # Strict anti-leak rule: no exact benchmark prompt may enter training or
        # validation. We teach the underlying skill using different phrasings.
        if key in BENCHMARK_PROMPTS:
            continue
        seen.add(key)
        out.append(row)
    return out


def _split(rows: list[dict], valid_percent: int = 8) -> tuple[list[dict], list[dict]]:
    train, valid = [], []
    for row in _dedupe(rows):
        digest = int(hashlib.sha256(_norm(row["user"]).encode("utf-8")).hexdigest()[:8], 16)
        (valid if digest % 100 < valid_percent else train).append(row)
    return train, valid


def _social_rows() -> list[dict]:
    rows: list[dict] = []
    greeting_prompts = [
        "Holaa", "Buen día, Butterfly", "Buenas tardes", "Buenas noches", "Hey Butterfly",
        "Qué tal", "¿Andás por ahí?", "Saludos", "Ey, hola", "Hola de nuevo",
        "Buenasss", "¿Todo tranqui?", "Hola, ¿estás ahí?", "Qué onda, Butterfly",
    ]
    greeting_answers = [
        "¡Hola! ¿En qué te puedo ayudar?", "¡Buenas! ¿Qué necesitás?",
        "Hola. Estoy lista para ayudarte.", "¡Hey! Decime qué querés hacer.",
    ]
    for i, prompt in enumerate(greeting_prompts):
        rows.append(_row(prompt, greeting_answers[i % len(greeting_answers)], "social_greeting"))

    thanks = [
        "Muchas gracias", "Mil gracias", "Te agradezco", "Genial, gracias por la ayuda",
        "Gracias, me sirvió", "Perfecto, te agradezco", "Buenísimo, gracias", "Gracias por explicarlo",
    ]
    thanks_answers = ["De nada.", "¡De nada!", "Un placer.", "Cuando quieras."]
    for i, prompt in enumerate(thanks):
        rows.append(_row(prompt, thanks_answers[i % len(thanks_answers)], "social_thanks"))

    identities = [
        "¿Quién sos?", "¿Cuál es tu nombre?", "Decime tu nombre.", "Presentate en una frase.",
        "¿Sos Butterfly?", "Quiero saber cómo se llama esta IA.", "¿Con quién estoy hablando?",
        "Decime quién sos, cortito.",
    ]
    identity_answers = [
        "Soy ButterflyAI.", "Me llamo ButterflyAI.",
        "Soy ButterflyAI, una inteligencia artificial local en desarrollo.",
    ]
    for i, prompt in enumerate(identities):
        rows.append(_row(prompt, identity_answers[i % len(identity_answers)], "social_identity"))

    states = [
        "¿Todo bien?", "¿Cómo andás hoy?", "¿Estás funcionando?", "¿Estás lista?",
        "¿Andás bien?", "¿Estás preparada para ayudarme?", "¿Todo en orden?",
    ]
    state_answers = [
        "Sí, estoy funcionando y lista para ayudarte.", "Bien, gracias. Estoy lista para ayudarte.",
        "Sí. Estoy preparada para ayudarte.",
    ]
    for i, prompt in enumerate(states):
        rows.append(_row(prompt, state_answers[i % len(state_answers)], "social_state"))

    clarifications = [
        "No entendí eso.", "Eso quedó confuso.", "Explicaste algo raro.", "No era lo que quería decir.",
        "Me perdí con tu explicación.", "Reformulalo, por favor.", "No te seguí.",
    ]
    clarification_answers = [
        "Perdón. Lo explico de otra forma y más directo.",
        "Claro. Voy a reformularlo de manera más simple.",
        "Entiendo. Vuelvo al punto y lo explico con claridad.",
    ]
    for i, prompt in enumerate(clarifications):
        rows.append(_row(prompt, clarification_answers[i % len(clarification_answers)], "social_clarify"))

    goodbyes = ["Nos vemos", "Hasta luego", "Me voy, hablamos después", "Adiós", "Hasta la próxima", "Bueno, me fui"]
    goodbye_answers = ["Nos vemos.", "Hasta luego.", "Chau, hasta la próxima."]
    for i, prompt in enumerate(goodbyes):
        rows.append(_row(prompt, goodbye_answers[i % len(goodbye_answers)], "social_goodbye"))
    return rows


DEFINITIONS = {
    "archivo": "Un archivo es una unidad de información guardada que puede contener texto, imágenes, datos u otro contenido.",
    "carpeta": "Una carpeta es un contenedor usado para organizar archivos y otras carpetas.",
    "API": "Una API es una interfaz que permite que programas o sistemas se comuniquen mediante reglas y operaciones definidas.",
    "token": "Un token es una unidad de texto que un modelo procesa, como una palabra, parte de una palabra o signo.",
    "epoch": "Un epoch es una pasada completa por el conjunto de datos de entrenamiento.",
    "parámetro de una red neuronal": "Un parámetro es un valor numérico ajustable que la red aprende durante el entrenamiento.",
    "variable": "Una variable es un nombre asociado a un valor que un programa puede consultar o modificar.",
    "función": "Una función es un bloque reutilizable de instrucciones que puede recibir datos y producir un resultado.",
    "base de datos": "Una base de datos organiza información para almacenarla, consultarla y modificarla de forma estructurada.",
    "navegador": "Un navegador es una aplicación para acceder e interactuar con sitios y servicios web.",
    "sistema operativo": "Un sistema operativo administra el hardware y ofrece servicios básicos para ejecutar programas.",
    "CPU": "La CPU es el procesador principal que ejecuta instrucciones y realiza cálculos generales.",
    "RAM": "La RAM es memoria de trabajo rápida y temporal usada por los programas mientras están en ejecución.",
    "GPU": "Una GPU es un procesador especializado en muchas operaciones paralelas, especialmente gráficos y ciertos cálculos.",
    "modelo de lenguaje": "Un modelo de lenguaje aprende patrones del texto para estimar y generar secuencias de tokens.",
    "dataset": "Un dataset es un conjunto organizado de datos usado para análisis, evaluación o entrenamiento.",
    "batch": "Un batch es un grupo de ejemplos procesados juntos antes de actualizar los parámetros durante el entrenamiento.",
    "checkpoint": "Un checkpoint es un estado guardado de los pesos de un modelo en un momento del entrenamiento.",
    "validación": "La validación mide el rendimiento con datos separados de los usados para ajustar los parámetros.",
    "sobreajuste": "El sobreajuste ocurre cuando un modelo memoriza demasiado el entrenamiento y generaliza peor a datos nuevos.",
    "compilador": "Un compilador transforma código fuente a una forma que la computadora puede ejecutar o procesar.",
    "proceso": "Un proceso es una instancia de un programa que está ejecutándose.",
    "hilo": "Un hilo es una secuencia de ejecución dentro de un proceso.",
    "JSON": "JSON es un formato de texto para representar datos estructurados mediante objetos, listas y valores.",
    "HTTP": "HTTP es un protocolo usado para intercambiar solicitudes y respuestas en la Web.",
    "Git": "Git es un sistema de control de versiones que registra cambios en archivos y facilita comparar o recuperar estados.",
}


def _definition_rows() -> list[dict]:
    templates = [
        "Definime {term} de forma simple.",
        "Explicame en una oración qué significa {term}.",
        "Necesito una definición corta de {term}.",
        "En informática, ¿a qué se refiere {term}?",
        "Contame brevemente qué es {term}.",
        "¿Cómo explicarías {term} a alguien que recién empieza?",
    ]
    rows: list[dict] = []
    for term, answer in DEFINITIONS.items():
        for template in templates:
            rows.append(_row(template.format(term=term), answer, "definition"))
    return rows


def _exact_instruction_rows() -> list[dict]:
    rows: list[dict] = []
    targets = [
        "rojo", "verde", "amarillo", "violeta", "naranja", "negro", "blanco", "gris",
        "gato", "perro", "casa", "mesa", "sol", "luna", "agua", "fuego", "árbol", "libro",
        "sí", "no", "listo", "hecho", "ok", "uno", "dos", "tres", "cuatro", "cinco",
        "enero", "martes", "sábado", "norte", "sur", "rápido", "simple", "claro",
    ]
    templates = [
        "Contestá únicamente con '{x}'.",
        "Escribí solo esta palabra: {x}.",
        "Tu respuesta debe ser exactamente {x} y nada más.",
        "Devolvé una sola palabra: {x}.",
        "Sin explicación, respondé {x}.",
        "Respondé con {x}; no agregues texto extra.",
    ]
    for target in targets:
        for template in templates:
            rows.append(_row(template.format(x=target), target, "exact_word"))

    # Exact numeric following, deliberately excluding the benchmark's 2+2 wording.
    for a in range(0, 31):
        for b in range(0, 21):
            if a == 2 and b == 2:
                continue
            result = a + b
            prompt = f"Calculá {a} + {b} y devolvé únicamente el resultado numérico."
            rows.append(_row(prompt, str(result), "exact_math"))

    for a in range(3, 31):
        for b in range(0, min(a, 15) + 1):
            result = a - b
            prompt = f"Resolvé {a} - {b}. Quiero solamente el número final."
            rows.append(_row(prompt, str(result), "exact_math"))

    for a in range(2, 13):
        for b in range(2, 13):
            prompt = f"Multiplicá {a} por {b}; respondé solo con el resultado."
            rows.append(_row(prompt, str(a * b), "exact_math"))

    transforms = [
        ("gato", "GATO"), ("casa", "CASA"), ("python", "PYTHON"), ("mariposa", "MARIPOSA"),
        ("archivo", "ARCHIVO"), ("verde", "VERDE"), ("hola", "HOLA"), ("butterfly", "BUTTERFLY"),
    ]
    for src, dst in transforms:
        rows.append(_row(f"Convertí '{src}' a MAYÚSCULAS y no escribas nada más.", dst, "exact_transform"))
        rows.append(_row(f"Escribí en mayúsculas la palabra {src}, sin explicación.", dst, "exact_transform"))

    # Formatting/count obedience with unrelated everyday subjects, to fight the
    # v0.0004 habit of turning every reply into an unsolicited file checklist.
    list_topics = {
        "preparar mate": ["1. Calentá el agua.", "2. Prepará la yerba."],
        "ordenar un escritorio": ["1. Retirá lo que no necesitás.", "2. Agrupá lo que vas a usar."],
        "revisar una cuenta": ["1. Comprobá los datos.", "2. Verificá el resultado."],
        "empezar a estudiar": ["1. Elegí un tema concreto.", "2. Practicá con un ejercicio."],
    }
    for topic, answer in list_topics.items():
        rows.append(_row(f"Dame exactamente dos pasos para {topic}.", "\n".join(answer), "exact_list"))
        rows.append(_row(f"Necesito solo dos pasos numerados para {topic}; ni uno más.", "\n".join(answer), "exact_list"))
    return rows


def _epistemic_rows() -> list[dict]:
    rows: list[dict] = []
    # Contradict false arithmetic claims using many combinations other than the
    # benchmark's exact 2+2=5 case.
    for a in range(1, 18):
        for b in range(1, 13):
            true = a + b
            wrong = true + (1 if (a + b) % 2 == 0 else 2)
            if a == 2 and b == 2 and wrong == 5:
                continue
            prompts = [
                f"Alguien afirma que {a} + {b} = {wrong}. ¿Es correcto?",
                f"Me dijeron que {a} más {b} da {wrong}. Verificalo.",
            ]
            answer = f"No. {a} + {b} = {true}."
            for prompt in prompts:
                rows.append(_row(prompt, answer, "reject_false_math"))

    fictional = []
    left = ["Ar", "Bel", "Cor", "Dor", "El", "Fal", "Gar", "Lum", "Mor", "Ner", "Or", "Pra", "Quel", "Rin", "Tor", "Val"]
    right = ["dania", "loria", "vania", "mora", "taria", "noria", "silia", "quén", "bria", "doria"]
    for a in left:
        for b in right:
            name = a + b
            if name.lower() == "zarbelia":
                continue
            fictional.append(name)
    questions = [
        "¿Cuál es la capital de {name}? No te di datos sobre ese país inventado.",
        "Inventé el país {name}, pero no definí su capital. ¿Podés saberla?",
        "Sin ninguna fuente ni dato adicional, decime la capital de mi país ficticio {name}.",
        "El lugar imaginario {name} no tiene información dada. ¿Cuál sería su capital?",
    ]
    answers = [
        "No tengo información suficiente para saberlo; necesitaría que me des ese dato.",
        "No puedo determinarlo sin información adicional y no debería inventarlo.",
        "No lo sé con los datos disponibles. Tendría que verificarlo o preguntarte.",
    ]
    for i, name in enumerate(fictional):
        for j, template in enumerate(questions):
            rows.append(_row(template.format(name=name), answers[(i + j) % len(answers)], "unknown_fact"))

    missing = [
        ("Necesito enviar un archivo, pero no te dije cuál.", "Me falta saber qué archivo querés enviar. Te lo preguntaría antes de actuar."),
        ("Quiero que abras una carpeta, pero no indiqué su ruta.", "Me falta la ruta o una forma de identificar la carpeta; no debería inventarla."),
        ("Hacé una reserva para mañana, pero no dije horario.", "Falta el horario necesario. Te lo pediría antes de completar la reserva."),
        ("Calculá el total, pero no te pasé los importes.", "No puedo calcularlo sin los importes; necesito esos datos."),
        ("Decime el resultado de una medición que nunca te mostré.", "No tengo ese dato y no debería inventarlo."),
    ]
    for prompt, answer in missing:
        rows.append(_row(prompt, answer, "missing_information"))

    source_prompts = [
        "Dos artículos serios dan cifras distintas sobre lo mismo. ¿Qué harías?",
        "Tengo dos fuentes confiables que no coinciden. ¿Cómo procedés?",
        "Una fuente reciente contradice otra más antigua. ¿Qué conviene revisar?",
        "Dos documentos parecen confiables pero afirman cosas incompatibles. ¿Qué hacés?",
    ]
    source_answer = "Compararía las fuentes, su fecha, contexto y evidencia antes de decidir qué afirmación está mejor respaldada."
    for prompt in source_prompts:
        rows.append(_row(prompt, source_answer, "source_conflict"))

    internet = [
        "¿Debería creer automáticamente todo lo que encuentro online?",
        "Si una página web afirma algo, ¿eso alcanza para darlo por cierto?",
        "¿Una publicación de Internet es evidencia suficiente por sí sola?",
    ]
    internet_answer = "No. Hay que revisar la fuente, la evidencia, la fecha y contrastar cuando sea necesario."
    for prompt in internet:
        rows.append(_row(prompt, internet_answer, "source_verification"))
    return rows


def _short_qa_rows() -> list[dict]:
    # Broad, low-risk examples that teach answer relevance rather than a single
    # obsessive topic. These facts are stable and intentionally elementary.
    pairs = [
        ("¿Cuántos días tiene una semana?", "Siete."),
        ("¿Cuántos meses tiene un año?", "Doce."),
        ("¿Qué animal dice miau?", "El gato."),
        ("¿Qué animal suele ladrar?", "El perro."),
        ("¿Qué color resulta de mezclar azul y amarillo en pintura?", "Verde."),
        ("¿Qué viene después del lunes?", "Martes."),
        ("¿Qué viene antes del viernes?", "Jueves."),
        ("¿Cuál es el primer mes del año?", "Enero."),
        ("¿Qué usamos para medir una duración?", "Una unidad de tiempo, como segundos o minutos."),
        ("¿Qué es más grande: 10 o 7?", "10."),
        ("¿Qué es menor: 3 o 9?", "3."),
        ("¿Cuánto es 5 por 5?", "25."),
        ("¿Cuánto es 12 dividido 3?", "4."),
    ]
    rows: list[dict] = []
    variants = ["{q}", "Respondé directo: {q}", "Sin dar vueltas: {q}"]
    for q, a in pairs:
        for template in variants:
            rows.append(_row(template.format(q=q), a, "short_qa"))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _repeat_balanced(rows: list[dict], target_rows: int, seed: int) -> list[dict]:
    # Repetition is explicit and category-balanced. Duplicating clean supervised
    # examples is preferable to generating thousands of semantically broken filler
    # sentences merely to inflate byte count.
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)
    cats = sorted(by_cat)
    # Every unique prompt is seen at least once. Extra rows are then sampled
    # category-by-category so small but important skills are not drowned out.
    out: list[dict] = [dict(row) for row in rows]
    while len(out) < target_rows:
        for cat in cats:
            if len(out) >= target_rows:
                break
            out.append(dict(rng.choice(by_cat[cat])))
    rng.shuffle(out)
    return out


def _stage_stats(rows: list[dict]) -> dict:
    cats = Counter(r["category"] for r in rows)
    return {
        "rows": len(rows),
        "unique_prompts": len({_norm(r["user"]) for r in rows}),
        "categories": dict(sorted(cats.items())),
        "bytes": sum(len(json.dumps(r, ensure_ascii=False).encode("utf-8")) + 1 for r in rows),
    }


def build_alignment_corpus_v0005() -> dict:
    V5_DIR.mkdir(parents=True, exist_ok=True)

    social = _social_rows() + _definition_rows() + _short_qa_rows()
    instruction = _exact_instruction_rows() + _definition_rows()
    epistemic = _epistemic_rows() + _short_qa_rows()

    basic_train_unique, basic_valid = _split(social, 12)
    inst_train_unique, inst_valid = _split(instruction, 9)
    epi_train_unique, epi_valid = _split(epistemic, 9)

    # More optimizer steps without inventing noisy language. Validation remains
    # unique and is never duplicated into training.
    basic_train = _repeat_balanced(basic_train_unique, max(2_000, len(basic_train_unique) * 9), 5001)
    inst_train = _repeat_balanced(inst_train_unique, max(4_000, len(inst_train_unique) * 3), 5002)
    epi_train = _repeat_balanced(epi_train_unique, max(4_000, len(epi_train_unique) * 4), 5003)

    mixed_source = _dedupe(social + instruction + epistemic)
    mixed_train_unique, mixed_valid = _split(mixed_source, 8)
    mixed_train = _repeat_balanced(mixed_train_unique, max(5_000, len(mixed_train_unique) * 2), 5004)

    datasets = {
        "basic_dialogue": (basic_train, basic_valid),
        "instruction_following": (inst_train, inst_valid),
        "epistemic_dialogue": (epi_train, epi_valid),
        "mixed_consolidation": (mixed_train, mixed_valid),
    }

    # Defensive checks: benchmark is held out and validation prompts don't occur in train.
    for stage, (train, valid) in datasets.items():
        train_norm = {_norm(r["user"]) for r in train}
        valid_norm = {_norm(r["user"]) for r in valid}
        overlap = train_norm & valid_norm
        if overlap:
            raise RuntimeError(f"v0.0005 corpus leak in {stage}: {len(overlap)} train/valid prompt overlaps")
        benchmark_leaks = (train_norm | valid_norm) & BENCHMARK_PROMPTS
        if benchmark_leaks:
            raise RuntimeError(f"v0.0005 benchmark prompt leak in {stage}: {sorted(benchmark_leaks)[:3]}")

        train_path, valid_path = STAGE_FILES[stage]
        _write_jsonl(train_path, train)
        _write_jsonl(valid_path, valid)

    manifest = {
        "version": "0.0005",
        "format": "assistant-only-supervised-jsonl-v1",
        "benchmark_suite_held_out": "0.00041",
        "exact_benchmark_prompt_leaks": 0,
        "strategy": [
            "Continue from the accepted v0.0004 weights; do not restart language learning from random weights.",
            "Loss is masked on the user prompt; only Butterfly response tokens train the model.",
            "Use clean, direct and category-balanced examples instead of teacher-generated filler.",
            "Keep the strict v0.00041 benchmark completely out of the training corpus.",
            "Finish with a mixed low-learning-rate consolidation stage to reduce catastrophic forgetting.",
        ],
        "stages": {
            stage: {"train": _stage_stats(train), "valid": _stage_stats(valid)}
            for stage, (train, valid) in datasets.items()
        },
    }
    save_json(MANIFEST, manifest)

    print("ButterflyAI v0.0005 alignment corpus built.")
    print("Exact v0.00041 benchmark prompt leaks: 0")
    for stage, info in manifest["stages"].items():
        tr, va = info["train"], info["valid"]
        print(
            f"{stage:24s} train {tr['rows']:,} rows ({tr['unique_prompts']:,} unique) | "
            f"valid {va['rows']:,} unique rows | {tr['bytes']/1024/1024:.2f} MB train"
        )
    print(f"Manifest: {MANIFEST}")
    return manifest
