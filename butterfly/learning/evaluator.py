from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata
from typing import Any

from ..generation import generate
from ..epistemic.engine import EpistemicEngine

BENCHMARK_SUITE_VERSION = "0.00042"

# These values are deliberately reserved for benchmark-only exact-copy cases.
# The v0.00051 corpus builder imports them and refuses to train on them.
BENCHMARK_RESERVED_EXACT_TARGETS = {"azul", "nube", "xq7"}
BENCHMARK_RESERVED_MATH = {
    ("+", 2, 2),
    ("+", 7, 6),
    ("-", 14, 9),
    ("*", 4, 3),
}
BENCHMARK_RESERVED_FALSE_MATH = {
    ("+", 2, 2, 5),
    ("+", 7, 8, 16),
}
BENCHMARK_RESERVED_FICTIONAL = {"zarbelia", "ormavia"}


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _norm(text: str) -> str:
    text = _strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9+\-*/?.! ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text)


def _contains(text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(text)


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(_contains(text, phrase) for phrase in phrases)


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
    if alpha_space < 0.38:
        score -= 0.20
    return max(0.0, min(1.0, score)) * _repetition_score(text)


def _sentence_count(text: str) -> int:
    chunks = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
    return len(chunks)


def _list_item_count(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line)) for line in lines)


def _semantic_score(answer: str, case: dict[str, Any]) -> float:
    validator = case.get("validator")
    norm = _norm(answer)

    if validator == "greeting":
        # A greeting is about intent, not one magic word. This deliberately accepts
        # natural replies such as "¡Hey! Decime qué querés hacer.".
        markers = ["hola", "buenas", "buen dia", "hey", "ey", "que tal", "saludos"]
        return 1.0 if _contains_any(answer, markers) else 0.0

    if validator == "thanks":
        markers = ["de nada", "no hay de que", "un placer", "cuando quieras", "por nada"]
        return 1.0 if _contains_any(answer, markers) else 0.0

    if validator == "identity":
        return 1.0 if "butterfly" in norm else 0.0

    if validator == "state":
        markers = [
            "bien", "lista", "listo", "funcionando", "preparada", "preparado",
            "operativa", "operativo", "todo bien", "aca", "aqui",
        ]
        return 1.0 if _contains_any(answer, markers) else 0.0

    if validator == "false_math":
        correct = str(case["correct_result"])
        neg = ["no", "incorrect", "falso", "equivoc", "no es correcto"]
        has_neg = _contains_any(answer, neg)
        has_correct = bool(re.search(rf"(?<!\d){re.escape(correct)}(?!\d)", norm))
        return 1.0 if has_neg and has_correct else (0.5 if has_correct else 0.0)

    if validator == "unknown":
        uncertainty = ["no se", "no lo se", "no puedo saber", "no tengo", "desconozco", "no puedo determinar"]
        missing = ["informacion", "dato", "datos", "verificar", "preguntar", "fuente"]
        return (float(_contains_any(answer, uncertainty)) + float(_contains_any(answer, missing))) / 2.0

    exact = case.get("exact_answer")
    if exact is not None:
        # Exact means exact: no quotes, explanation or decorative punctuation.
        return 1.0 if answer.strip().casefold() == str(exact).strip().casefold() else 0.0

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

    if case.get("direct") and nwords > 0 and answer.lstrip().startswith(("1.", "1)", "- ", "* ")):
        score -= 0.25
        reasons.append("not direct")

    return max(0.0, min(1.0, score)), reasons


COMMON_FORBIDDEN = [
    "archivo", "carpeta", "codigo", "sistema operativo", "compilar", "excel",
    "proyecto", "si te pido", "que preguntas o comprobaciones",
]

# v0.00042 deliberately contains multiple surface forms for the same capability:
# no punctuation, no accents, colloquial spelling, no quotation marks, and close
# contrastive intents. Passing one memorized wording is no longer enough.
CASES: list[dict[str, Any]] = [
    # --- Conversation / contrastive social intent ---
    {"id":"hello","category":"conversation","prompt":"Hola","validator":"greeting","critical":True,"direct":True,"no_list":True,"max_words":18},
    {"id":"hello_casual","category":"conversation","prompt":"buenas q onda","validator":"greeting","critical":True,"robust":True,"direct":True,"no_list":True,"max_words":20},
    {"id":"thanks_casual","category":"conversation","prompt":"graciass me re sirvio","validator":"thanks","critical":True,"robust":True,"direct":True,"no_list":True,"max_words":18},
    {"id":"identity_plain","category":"conversation","prompt":"como te llamas","validator":"identity","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":24},
    {"id":"identity_alt","category":"conversation","prompt":"quien sos vos","validator":"identity","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":28},
    {"id":"how_are_you_plain","category":"conversation","prompt":"como estas","validator":"state","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":28},
    {"id":"how_are_you_alt","category":"conversation","prompt":"todo bien por ahi","validator":"state","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":28},
    {"id":"clarify","category":"conversation","prompt":"no entendi nada explicalo de nuevo","required_groups":[["perdon","discul","aclar","explic","otra forma","reformular"]],"robust":True,"no_list":True,"max_words":40},
    {"id":"goodbye","category":"conversation","prompt":"bueno me fui nos vemos","required_groups":[["chau","adios","hasta luego","nos vemos","hasta la proxima"]],"robust":True,"direct":True,"no_list":True,"max_words":20},

    # --- Basic Spanish / comprehension with casual input ---
    {"id":"file_casual","category":"comprehension","prompt":"archivo q es explicame facil","required_groups":[["archivo"],["informacion","contenido","datos"],["guard","almacen","conten"]],"robust":True,"max_words":55},
    {"id":"folder_casual","category":"comprehension","prompt":"que seria una carpeta en la pc","required_groups":[["carpeta"],["archivo"],["organiz","conten","guard","agrup"]],"robust":True,"max_words":55},
    {"id":"api_casual","category":"comprehension","prompt":"explicame api sin vueltas","required_groups":[["api","interfaz"],["programa","aplicacion","software","sistema"],["comunic","conect","interact","solicitud"]],"robust":True,"max_words":65},
    {"id":"parameter","category":"comprehension","prompt":"q significa parametro en una red neuronal","required_groups":[["parametro","peso"],["valor","numero"],["ajust","aprend","entren"]],"robust":True,"max_words":75},
    {"id":"token","category":"comprehension","prompt":"token en un modelo de ia q es","required_groups":[["token"],["texto","palabra","pieza","parte","fragmento"]],"robust":True,"max_words":65},
    {"id":"epoch","category":"comprehension","prompt":"epoch q significa cuando entrenas una ia","required_groups":[["epoch","epoca"],["datos","dataset","entrenamiento","conjunto"],["vez","pasada","recorrido","vuelta","completo"]],"robust":True,"max_words":80},
    {"id":"ram","category":"comprehension","prompt":"para q sirve la ram","required_groups":[["ram","memoria"],["temporal","trabajo","programa","ejecucion"]],"robust":True,"max_words":55},

    # --- Exact binding: benchmark targets are never used by v0.00051 train/valid ---
    {"id":"exact_blue","category":"instruction","prompt":"responde solamente con la palabra azul","exact_answer":"azul","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"exact_cloud","category":"instruction","prompt":"sin comillas ni explicacion escribi nube","exact_answer":"nube","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"exact_code","category":"instruction","prompt":"tu respuesta tiene que ser solo XQ7 nada mas","exact_answer":"XQ7","robust":True,"contrastive":True,"max_words":1},
    {"id":"one_sentence_file","category":"instruction","prompt":"en una sola oracion decime que es un archivo","required_groups":[["archivo"],["informacion","datos","contenido"]],"robust":True,"max_sentences":1,"max_words":40},
    {"id":"two_steps","category":"instruction","prompt":"dame exactamente dos pasos para revisar por que un proyecto no compila","required_groups":[["error","compil"],["codigo","archivo","dependencia","configuracion"]],"robust":True,"expected_items":2,"max_words":70},
    {"id":"missing_data","category":"instruction","prompt":"te falta un dato clave para hacer la tarea q haces","required_groups":[["pregunt","pedir","solicitar","verific"],["no invent","no asumir","falta","desconozco","no se"]],"robust":True,"max_words":60},
    {"id":"short_summary","category":"instruction","prompt":"en menos de 10 palabras para que sirve una carpeta","required_groups":[["carpeta"],["organiz","guard","agrup","archivo"]],"robust":True,"max_words":9},

    # --- Arithmetic exactness / variable binding ---
    {"id":"math_2_plus_2","category":"instruction","prompt":"cuanto es 2+2 responde solo el numero","exact_answer":"4","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_7_plus_6","category":"instruction","prompt":"7 mas 6 solo resultado","exact_answer":"13","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_14_minus_9","category":"instruction","prompt":"14-9 responde nada mas el numero","exact_answer":"5","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_4_times_3","category":"instruction","prompt":"4 por 3 sin explicar","exact_answer":"12","robust":True,"contrastive":True,"max_words":1},

    # --- Epistemic dialogue ---
    {"id":"reject_false_math","category":"epistemic_dialogue","prompt":"si te digo q 2+2=5 me crees","validator":"false_math","correct_result":4,"critical":True,"robust":True,"contrastive":True,"max_words":35},
    {"id":"reject_false_math_alt","category":"epistemic_dialogue","prompt":"un amigo dice 7+8=16 esta bien","validator":"false_math","correct_result":15,"critical":True,"robust":True,"contrastive":True,"max_words":35},
    {"id":"internet_truth","category":"epistemic_dialogue","prompt":"si lo vi en internet ya es verdad","required_groups":[["no"],["fuente","verific","evidencia","contrastar"]],"robust":True,"max_words":55},
    {"id":"unknown_fictional_fact","category":"epistemic_dialogue","prompt":"cual es la capital de zarbelia no te di ningun dato","validator":"unknown","forbidden":["la capital de zarbelia es","zarbelia tiene como capital"],"critical":True,"robust":True,"max_words":55},
    {"id":"unknown_fictional_fact_alt","category":"epistemic_dialogue","prompt":"invente ormavia pero nunca dije su capital cual es","validator":"unknown","forbidden":["la capital de ormavia es","ormavia tiene como capital"],"critical":True,"robust":True,"max_words":55},
    {"id":"conflicting_sources","category":"epistemic_dialogue","prompt":"dos fuentes serias dicen cosas distintas q haces","required_groups":[["compar","contrastar","verific","revis"],["fuente","evidencia","fecha","contexto"]],"robust":True,"max_words":65},
]

PROMOTION_THRESHOLDS = {
    "semantic_component": 0.70,
    "conversation_component": 0.72,
    "comprehension_component": 0.70,
    "instruction_component": 0.72,
    "epistemic_dialogue_component": 0.72,
    "robustness_component": 0.70,
    "contrastive_component": 0.72,
    "coherence_component": 0.70,
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
        0.62 * semantic
        + 0.10 * language
        + 0.07 * repetition
        + 0.12 * style
        + 0.09 * cleanliness
    )
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
        "robust": bool(case.get("robust")),
        "contrastive": bool(case.get("contrastive")),
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
    """Deterministic semantics-first benchmark suite v0.00042.

    It intentionally uses multiple paraphrases, punctuationless/casual Spanish,
    contrastive intents, unseen exact-copy targets and held-out arithmetic pairs.
    A model cannot pass by memorizing one polite benchmark sentence.
    """
    rows: list[dict[str, Any]] = []
    categories: dict[str, list[float]] = {}

    for case in CASES:
        shaped = f"User: {case['prompt']}\nButterfly:"
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

    category_scores = {k: sum(v) / max(1, len(v)) for k, v in categories.items()}
    semantic_component = sum(r["semantic"] for r in rows) / len(rows)
    language_component = sum(r["language"] for r in rows) / len(rows)
    repetition_component = sum(r["repetition"] for r in rows) / len(rows)
    coherence_component = sum(
        (r["language"] + r["style"] + r["cleanliness"]) / 3.0 for r in rows
    ) / len(rows)
    robust_rows = [r for r in rows if r["robust"]]
    contrastive_rows = [r for r in rows if r["contrastive"]]
    robustness_component = sum(r["score"] for r in robust_rows) / max(1, len(robust_rows))
    contrastive_component = sum(r["score"] for r in contrastive_rows) / max(1, len(contrastive_rows))

    engine = EpistemicEngine()
    epi_tests = [
        ("2 + 2 = 4", "VERIFIED"), ("2 + 2 = 5", "CONTRADICTED"),
        ("10 / 2 = 5", "VERIFIED"), ("3 * 7 = 20", "CONTRADICTED"),
        ("9 - 4 = 5", "VERIFIED"), ("6 * 6 = 35", "CONTRADICTED"),
        ("12 / 3 = 4", "VERIFIED"), ("7 + 8 = 14", "CONTRADICTED"),
    ]
    epistemic_engine_component = sum(
        engine.verify(claim).status.value == expected for claim, expected in epi_tests
    ) / len(epi_tests)

    critical_failures = [r["id"] for r in rows if r["critical"] and not r["critical_pass"]]
    critical_total = sum(r["critical"] for r in rows)
    critical_pass_rate = (
        (critical_total - len(critical_failures)) / critical_total if critical_total else 1.0
    )

    overall = (
        0.18 * semantic_component
        + 0.15 * category_scores.get("conversation", 0.0)
        + 0.14 * category_scores.get("comprehension", 0.0)
        + 0.15 * category_scores.get("instruction", 0.0)
        + 0.12 * category_scores.get("epistemic_dialogue", 0.0)
        + 0.08 * robustness_component
        + 0.06 * contrastive_component
        + 0.06 * coherence_component
        + 0.03 * repetition_component
        + 0.03 * epistemic_engine_component
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
        "robustness_component": robustness_component,
        "contrastive_component": contrastive_component,
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
        "score", "semantic_component", "language_component", "conversation_component",
        "comprehension_component", "instruction_component", "epistemic_dialogue_component",
        "epistemic_engine_component", "robustness_component", "contrastive_component",
        "coherence_component", "repetition_component", "critical_pass_rate",
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

    print("\nCritical / robustness samples:")
    for row in [r for r in metrics["cases"] if r["critical"]]:
        status = "PASS" if row["critical_pass"] else "FAIL"
        print(f"[{status}] You > {row['prompt']}\nButterfly > {row['answer']}\n")

    print("Additional comprehension samples:")
    for row in [r for r in metrics["cases"] if r["category"] == "comprehension"][:5]:
        print(f"You > {row['prompt']}\nButterfly > {row['answer']}\n")


def save_benchmark(path: Path, metrics: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
