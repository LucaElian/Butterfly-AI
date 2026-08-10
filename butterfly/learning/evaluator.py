from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata
from typing import Any

from ..generation import generate
from ..epistemic.engine import EpistemicEngine

BENCHMARK_SUITE_VERSION = "0.00041"


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9ñáéíóúü+\-*/?.! ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text)


def _contains(text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(text)


def _repetition_score(text: str) -> float:
    words = re.findall(r"[a-záéíóúüñ0-9]+", text.lower())
    if len(words) < 8:
        return 1.0
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    unique = len(set(grams)) / max(1, len(grams))
    bad_char = bool(re.search(r"(.{2,6})\1\1", text.lower()))
    bad_word = bool(re.search(r"\b([a-záéíóúüñ]{2,8})\s+\1\s+\1\b", text.lower()))
    penalty = (0.35 if bad_char else 0.0) + (0.20 if bad_word else 0.0)
    return max(0.0, min(1.0, unique - penalty))


def _language_quality(text: str) -> float:
    if not text.strip() or "�" in text:
        return 0.0
    words = _words(text)
    if not words:
        return 0.1
    bizarre = 0
    vowels = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
    for word in words:
        if len(word) > 28:
            bizarre += 1
        elif len(word) > 7 and not any(ch in vowels for ch in word):
            bizarre += 1
        elif re.search(r"(.{2,4})\1\1", word.lower()):
            bizarre += 1
    printable = sum(ch.isprintable() for ch in text) / max(1, len(text))
    alpha_space = sum(ch.isalpha() or ch.isspace() for ch in text) / max(1, len(text))
    score = 1.0 - bizarre / max(1, len(words))
    if printable < 0.98:
        score -= 0.25
    if alpha_space < 0.42:
        score -= 0.20
    return max(0.0, min(1.0, score)) * _repetition_score(text)


def _sentence_count(text: str) -> int:
    chunks = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
    return len(chunks)


def _list_item_count(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line)) for line in lines)


def _semantic_score(answer: str, case: dict[str, Any]) -> float:
    exact = case.get("exact_answer")
    if exact is not None:
        cleaned = _norm(answer).strip(" .!?\n\t")
        return 1.0 if cleaned == _norm(exact) else 0.0

    groups = case.get("required_groups", [])
    if not groups:
        return 1.0
    hits = 0
    for group in groups:
        if any(_contains(answer, phrase) for phrase in group):
            hits += 1
    return hits / len(groups)


def _cleanliness_score(answer: str, case: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 1.0

    forbidden = case.get("forbidden", [])
    forbidden_hits = [term for term in forbidden if _contains(answer, term)]
    if forbidden_hits:
        score -= min(0.75, 0.22 * len(forbidden_hits))
        reasons.append("topic contamination: " + ", ".join(forbidden_hits[:4]))

    artifacts = ["User:", "Butterfly:", "<END>", "### Instruction", "### Response"]
    artifact_hits = [x for x in artifacts if x.lower() in answer.lower()]
    if artifact_hits:
        score -= 0.55
        reasons.append("training-format leak")

    if re.search(r"\\\*|\\_", answer):
        score -= 0.18
        reasons.append("escaped markdown noise")

    if answer.count("**") >= 6:
        score -= 0.12
        reasons.append("excessive markdown")

    prompt = case.get("prompt", "")
    nprompt = _norm(prompt)
    nanswer = _norm(answer)
    if len(nprompt) >= 12 and nprompt in nanswer:
        score -= 0.20
        reasons.append("prompt echo")

    return max(0.0, min(1.0, score)), reasons


def _style_score(answer: str, case: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 1.0
    words = _words(answer)
    nwords = len(words)

    min_words = case.get("min_words")
    max_words = case.get("max_words")
    if min_words is not None and nwords < min_words:
        score -= 0.35
        reasons.append(f"too short ({nwords} words)")
    if max_words is not None and nwords > max_words:
        excess = nwords - max_words
        score -= min(0.75, 0.25 + 0.02 * excess)
        reasons.append(f"too long ({nwords}>{max_words} words)")

    if case.get("no_list") and _list_item_count(answer) > 0:
        score -= 0.45
        reasons.append("unnecessary list")

    max_sentences = case.get("max_sentences")
    if max_sentences is not None and _sentence_count(answer) > max_sentences:
        score -= 0.40
        reasons.append("too many sentences")

    expected_items = case.get("expected_items")
    if expected_items is not None:
        actual = _list_item_count(answer)
        if actual != expected_items:
            score -= 0.55
            reasons.append(f"expected {expected_items} list items, got {actual}")

    if case.get("direct") and nwords > 0:
        if answer.lstrip().startswith(("1.", "1)", "- ", "* ")):
            score -= 0.25
            reasons.append("not direct")

    return max(0.0, min(1.0, score)), reasons


CASSES_COMMON_FORBIDDEN = [
    "archivo",
    "carpeta",
    "codigo",
    "sistema operativo",
    "compilar",
    "excel",
    "proyecto",
    "datos",
    "si te pido",
    "que preguntas o comprobaciones",
]


CASES: list[dict[str, Any]] = [
    {
        "id": "hello",
        "category": "conversation",
        "prompt": "Hola",
        "required_groups": [["hola", "buenas", "buen día", "buen dia"]],
        "forbidden": CASSES_COMMON_FORBIDDEN,
        "critical": True,
        "direct": True,
        "no_list": True,
        "max_words": 18,
    },
    {
        "id": "hello_buenas",
        "category": "conversation",
        "prompt": "Buenas",
        "required_groups": [["hola", "buenas", "buen día", "buen dia"]],
        "forbidden": CASSES_COMMON_FORBIDDEN,
        "critical": True,
        "direct": True,
        "no_list": True,
        "max_words": 18,
    },
    {
        "id": "thanks",
        "category": "conversation",
        "prompt": "Gracias",
        "required_groups": [["de nada", "no hay de qué", "no hay de que", "un placer", "cuando quieras"]],
        "forbidden": CASSES_COMMON_FORBIDDEN,
        "critical": True,
        "direct": True,
        "no_list": True,
        "max_words": 18,
    },
    {
        "id": "identity",
        "category": "conversation",
        "prompt": "¿Cómo te llamás?",
        "required_groups": [["butterfly"]],
        "forbidden": CASSES_COMMON_FORBIDDEN,
        "critical": True,
        "direct": True,
        "no_list": True,
        "max_words": 22,
    },
    {
        "id": "how_are_you",
        "category": "conversation",
        "prompt": "¿Cómo estás?",
        "required_groups": [["bien", "lista", "funcionando", "preparada"]],
        "forbidden": CASSES_COMMON_FORBIDDEN,
        "critical": True,
        "direct": True,
        "no_list": True,
        "max_words": 24,
    },
    {
        "id": "clarify",
        "category": "conversation",
        "prompt": "¿Qué estás diciendo?",
        "required_groups": [["perdón", "perdon", "discul", "aclar", "explic", "me expresé", "me exprese"]],
        "forbidden": ["excel", "compilar", "si te pido"],
        "critical": False,
        "no_list": True,
        "max_words": 36,
    },
    {
        "id": "goodbye",
        "category": "conversation",
        "prompt": "Chau",
        "required_groups": [["chau", "adiós", "adios", "hasta luego", "nos vemos"]],
        "forbidden": CASSES_COMMON_FORBIDDEN,
        "critical": False,
        "direct": True,
        "no_list": True,
        "max_words": 18,
    },
    {
        "id": "did_not_understand",
        "category": "conversation",
        "prompt": "No entendí lo que dijiste.",
        "required_groups": [["explic", "aclar", "otra forma", "reformular", "de nuevo"]],
        "critical": False,
        "no_list": True,
        "max_words": 34,
    },
    {
        "id": "file",
        "category": "comprehension",
        "prompt": "Explica qué es un archivo.",
        "required_groups": [["archivo"], ["datos", "información", "informacion", "contenido"], ["guard", "almacen", "conten"]],
        "max_words": 55,
    },
    {
        "id": "folder",
        "category": "comprehension",
        "prompt": "¿Qué es una carpeta?",
        "required_groups": [["carpeta"], ["archivo", "archivos"], ["organiz", "conten", "guard", "agrup"]],
        "max_words": 55,
    },
    {
        "id": "api",
        "category": "comprehension",
        "prompt": "¿Qué es una API?",
        "required_groups": [["api", "interfaz"], ["programa", "aplicación", "aplicacion", "software", "sistema"], ["comunic", "conect", "interact", "solicitud"]],
        "max_words": 65,
    },
    {
        "id": "parameter",
        "category": "comprehension",
        "prompt": "¿Qué es un parámetro de una red neuronal?",
        "required_groups": [["parámetro", "parametro", "peso"], ["valor", "número", "numero"], ["ajust", "aprend", "entren"]],
        "max_words": 75,
    },
    {
        "id": "token",
        "category": "comprehension",
        "prompt": "¿Qué es un token en un modelo de lenguaje?",
        "required_groups": [["token"], ["texto", "palabra", "pieza", "parte", "fragmento"]],
        "max_words": 65,
    },
    {
        "id": "epoch",
        "category": "comprehension",
        "prompt": "En entrenamiento de IA, ¿qué significa epoch?",
        "required_groups": [["epoch", "época", "epoca"], ["datos", "dataset", "entrenamiento", "conjunto"], ["vez", "pasada", "recorrido", "vuelta", "completo"]],
        "max_words": 80,
    },
    {
        "id": "exact_blue",
        "category": "instruction",
        "prompt": "Respondé solamente con la palabra azul.",
        "exact_answer": "azul",
        "critical": True,
        "max_words": 1,
    },
    {
        "id": "one_sentence_file",
        "category": "instruction",
        "prompt": "Respondé en una sola oración: ¿qué es un archivo?",
        "required_groups": [["archivo"], ["datos", "información", "informacion", "contenido"]],
        "max_sentences": 1,
        "max_words": 40,
    },
    {
        "id": "two_steps",
        "category": "instruction",
        "prompt": "Dame exactamente dos pasos para revisar por qué un proyecto no compila.",
        "required_groups": [["error", "compil"], ["código", "codigo", "archivo", "dependencia", "configuración", "configuracion"]],
        "expected_items": 2,
        "max_words": 65,
    },
    {
        "id": "missing_data",
        "category": "instruction",
        "prompt": "Te falta un dato necesario para completar una tarea. ¿Qué deberías hacer?",
        "required_groups": [["pregunt", "pedir", "solicitar", "verific"], ["no invent", "no asumir", "falta", "desconozco", "no sé", "no se"]],
        "max_words": 60,
    },
    {
        "id": "short_summary",
        "category": "instruction",
        "prompt": "En menos de 12 palabras, explica para qué sirve una carpeta.",
        "required_groups": [["carpeta"], ["organiz", "guard", "agrup", "archivo"]],
        "max_words": 11,
    },
    {
        "id": "math_2_plus_2",
        "category": "epistemic_dialogue",
        "prompt": "¿Cuánto es 2 + 2? Respondé solo con el número.",
        "exact_answer": "4",
        "critical": True,
        "max_words": 1,
    },
    {
        "id": "reject_false_math",
        "category": "epistemic_dialogue",
        "prompt": "Si te digo que 2 + 2 = 5, ¿me creés?",
        "required_groups": [["no", "incorrect", "falso"], ["4"]],
        "critical": True,
        "max_words": 45,
    },
    {
        "id": "internet_truth",
        "category": "epistemic_dialogue",
        "prompt": "¿Todo lo que aparece en Internet es verdad?",
        "required_groups": [["no"], ["fuente", "verific", "evidencia", "contrastar"]],
        "max_words": 55,
    },
    {
        "id": "unknown_fictional_fact",
        "category": "epistemic_dialogue",
        "prompt": "¿Cuál es la capital del país ficticio Zarbelia? No te di ninguna información adicional.",
        "required_groups": [["no sé", "no se", "no puedo saber", "no tengo", "desconozco"], ["información", "informacion", "dato", "verific", "pregunt"]],
        "forbidden": ["la capital de zarbelia es", "zarbelia tiene como capital"],
        "critical": True,
        "max_words": 55,
    },
    {
        "id": "conflicting_sources",
        "category": "epistemic_dialogue",
        "prompt": "Dos fuentes confiables se contradicen. ¿Qué hacés?",
        "required_groups": [["compar", "contrastar", "verific", "revis"], ["fuente", "evidencia", "fecha", "contexto"]],
        "max_words": 65,
    },
]


PROMOTION_THRESHOLDS = {
    "semantic_component": 0.72,
    "conversation_component": 0.72,
    "comprehension_component": 0.62,
    "instruction_component": 0.68,
    "epistemic_dialogue_component": 0.70,
    "coherence_component": 0.65,
    "repetition_component": 0.85,
    "epistemic_engine_component": 0.99,
}


def _case_result(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    semantic = _semantic_score(answer, case)
    language = _language_quality(answer)
    repetition = _repetition_score(answer)
    style, style_reasons = _style_score(answer, case)
    cleanliness, clean_reasons = _cleanliness_score(answer, case)

    raw_score = (
        0.60 * semantic
        + 0.12 * language
        + 0.08 * repetition
        + 0.12 * style
        + 0.08 * cleanliness
    )
    # A response that happens to contain the expected keyword should not receive
    # a great score if it is rambling, contaminated by an unrelated topic, or
    # malformed.  This quality gate is the main correction over the v0.0004
    # benchmark, which over-rewarded superficial keyword hits.
    quality_gate = 0.35 + 0.65 * min(style, cleanliness)
    score = max(0.0, min(1.0, raw_score * quality_gate))

    critical_pass = True
    if case.get("critical"):
        critical_pass = semantic >= 0.999 and style >= 0.55 and cleanliness >= 0.55

    return {
        "id": case["id"],
        "category": case["category"],
        "prompt": case["prompt"],
        "answer": answer,
        "semantic": semantic,
        "language": language,
        "repetition": repetition,
        "style": style,
        "cleanliness": cleanliness,
        "score": score,
        "critical": bool(case.get("critical")),
        "critical_pass": critical_pass,
        "notes": style_reasons + clean_reasons,
    }


def _promotion_check(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    failures = metrics.get("critical_failures", [])
    if failures:
        reasons.append("critical failures: " + ", ".join(failures))
    for key, threshold in PROMOTION_THRESHOLDS.items():
        value = float(metrics.get(key, 0.0))
        if value < threshold:
            reasons.append(f"{key} {value:.4f} < {threshold:.4f}")
    return not reasons, reasons


def behavior_benchmark(model, tokenizer, max_new_tokens: int = 96):
    """Deterministic, semantics-first benchmark suite v0.00041.

    top_k=1 makes each answer reproducible. The suite intentionally gives much
    more weight to answering the actual question than to merely producing valid
    looking Spanish text.
    """
    rows: list[dict[str, Any]] = []
    categories: dict[str, list[float]] = {}

    for case in CASES:
        prompt = case["prompt"]
        shaped = f"User: {prompt}\nButterfly:"
        out = generate(
            model,
            shaped,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            repetition_penalty=1.25,
        )
        answer = out[len(shaped) :]
        for marker in ("<END>", "\nUser:"):
            if marker in answer:
                answer = answer.split(marker, 1)[0]
        answer = answer.strip()
        row = _case_result(answer, case)
        rows.append(row)
        categories.setdefault(case["category"], []).append(row["score"])

    category_scores = {
        key: sum(values) / max(1, len(values)) for key, values in categories.items()
    }
    semantic_component = sum(r["semantic"] for r in rows) / len(rows)
    language_component = sum(r["language"] for r in rows) / len(rows)
    repetition_component = sum(r["repetition"] for r in rows) / len(rows)
    coherence_component = sum(
        (r["language"] + r["style"] + r["cleanliness"]) / 3.0 for r in rows
    ) / len(rows)

    engine = EpistemicEngine()
    epi_tests = [
        ("2 + 2 = 4", "VERIFIED"),
        ("2 + 2 = 5", "CONTRADICTED"),
        ("10 / 2 = 5", "VERIFIED"),
        ("3 * 7 = 20", "CONTRADICTED"),
        ("9 - 4 = 5", "VERIFIED"),
        ("6 * 6 = 35", "CONTRADICTED"),
        ("12 / 3 = 4", "VERIFIED"),
        ("7 + 8 = 14", "CONTRADICTED"),
    ]
    epistemic_engine_component = sum(
        engine.verify(claim).status.value == expected for claim, expected in epi_tests
    ) / len(epi_tests)

    critical_failures = [
        row["id"] for row in rows if row["critical"] and not row["critical_pass"]
    ]
    critical_total = sum(row["critical"] for row in rows)
    critical_pass_rate = (
        (critical_total - len(critical_failures)) / critical_total if critical_total else 1.0
    )

    overall = (
        0.20 * semantic_component
        + 0.18 * category_scores.get("conversation", 0.0)
        + 0.16 * category_scores.get("comprehension", 0.0)
        + 0.16 * category_scores.get("instruction", 0.0)
        + 0.12 * category_scores.get("epistemic_dialogue", 0.0)
        + 0.08 * coherence_component
        + 0.05 * repetition_component
        + 0.05 * epistemic_engine_component
    )

    metrics: dict[str, Any] = {
        "suite_version": BENCHMARK_SUITE_VERSION,
        "score": overall,
        "semantic_component": semantic_component,
        "language_component": language_component,
        "conversation_component": category_scores.get("conversation", 0.0),
        "comprehension_component": category_scores.get("comprehension", 0.0),
        "instruction_component": category_scores.get("instruction", 0.0),
        "epistemic_dialogue_component": category_scores.get("epistemic_dialogue", 0.0),
        "epistemic_engine_component": epistemic_engine_component,
        "coherence_component": coherence_component,
        "repetition_component": repetition_component,
        "critical_pass_rate": critical_pass_rate,
        "critical_failures": critical_failures,
        "cases": rows,
    }
    eligible, reasons = _promotion_check(metrics)
    metrics["promotion_eligible"] = eligible
    metrics["promotion_blockers"] = reasons
    return metrics


def print_benchmark(metrics):
    keys = [
        "score",
        "semantic_component",
        "language_component",
        "conversation_component",
        "comprehension_component",
        "instruction_component",
        "epistemic_dialogue_component",
        "epistemic_engine_component",
        "coherence_component",
        "repetition_component",
        "critical_pass_rate",
    ]
    print(f"Benchmark suite                 : v{metrics.get('suite_version', '?')}")
    for key in keys:
        print(f"{key:32s}: {metrics.get(key, 0):.4f}")

    eligible = metrics.get("promotion_eligible", False)
    print(f"promotion_eligible              : {'YES' if eligible else 'NO'}")
    if not eligible:
        print("Promotion blockers:")
        for reason in metrics.get("promotion_blockers", []):
            print(f"  - {reason}")

    print("\nCritical/basic dialogue samples:")
    interesting = [r for r in metrics["cases"] if r["critical"]]
    for row in interesting:
        status = "PASS" if row["critical_pass"] else "FAIL"
        print(f"[{status}] You > {row['prompt']}\nButterfly > {row['answer']}\n")

    print("Additional comprehension samples:")
    for row in [r for r in metrics["cases"] if r["category"] == "comprehension"][:4]:
        print(f"You > {row['prompt']}\nButterfly > {row['answer']}\n")


def save_benchmark(path: Path, metrics: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
