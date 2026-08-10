from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata
from typing import Any

from ..generation import generate
from ..epistemic.engine import EpistemicEngine

BENCHMARK_SUITE_VERSION = "0.00045"

# Reserved benchmark-only values. The deliberate corpus builder imports these and
# refuses to place them in train OR validation. This keeps the final benchmark a
# genuine transfer test rather than a memorized lookup table.
BENCHMARK_RESERVED_EXACT_TARGETS = {
    "azul", "nube", "xq7", "cobre", "mate", "lz4", "r7k2",
}
BENCHMARK_RESERVED_MATH = {
    ("+", 2, 2),
    ("+", 7, 6),
    ("-", 14, 9),
    ("*", 4, 3),
    ("+", 8, 9),
    ("-", 17, 8),
    ("*", 6, 4),
    ("+", 12, 15),
    ("-", 20, 7),
    ("*", 3, 7),
}
BENCHMARK_RESERVED_FALSE_MATH = {
    ("+", 2, 2, 5),
    ("+", 7, 8, 16),
    ("+", 9, 6, 14),
    ("-", 12, 5, 8),
}
BENCHMARK_RESERVED_FICTIONAL = {"zarbelia", "ormavia", "veloria", "tarsenia"}


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_surface(text: str) -> str:
    """Normalization used by both the benchmark and anti-leak corpus builder.

    Accent/case/punctuation-only changes do not create a new benchmark surface.
    A few common casual abbreviations are normalized too, so `q` is not enough to
    sneak a held-out sentence into training.
    """
    text = _strip_accents(text.lower())
    text = re.sub(r"\bxq\b|\bpq\b", " porque ", text)
    text = re.sub(r"\bq\b", " que ", text)
    text = re.sub(r"\bmas\b", " + ", text)
    text = re.sub(r"\bmenos\b", " - ", text)
    text = re.sub(r"\bpor\b", " * ", text)
    text = re.sub(r"[^a-z0-9+\-*/ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm(text: str) -> str:
    text = _strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9+\-*/?.! ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def benchmark_surface_prompts() -> set[str]:
    return {normalize_surface(case["prompt"]) for case in CASES}


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
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
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
    return len([x for x in re.split(r"[.!?]+", text) if x.strip()])


def _list_item_count(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line)) for line in lines)



def _has_any_norm(answer: str, stems: list[str]) -> bool:
    norm = _norm(answer)
    return any(_norm(stem) in norm for stem in stems)



def _definition_is_noise(answer: str) -> bool:
    """Reject procedure/template leakage before semantic keyword checks."""
    norm = _norm(answer)
    if "�" in answer:
        return True
    if _list_item_count(answer) >= 2:
        return True
    bad = [
        "si te pido",
        "que preguntas o comprobaciones",
        "propone un plan",
        "propon un plan",
        "identifica el calculo",
        "verificar los datos",
        "verificar si hay errores",
        "te gustaria hacer",
        "quieres ayudarme",
        "revisar un archivo",
        "revisar una carpeta",
        "revisar un proyecto",
    ]
    return any(_norm(x) in norm for x in bad)


def _definition_score(answer: str, concept: str) -> float:
    """Judge concept meaning in prompt context while rejecting fluent-looking noise."""
    if _definition_is_noise(answer):
        return 0.0

    if concept == "file":
        has_content = _has_any_norm(
            answer, ["informacion", "contenido", "datos", "texto", "imagen", "documento"]
        )
        # Do not count the bare word "archivo" as evidence that the answer explained storage.
        has_storage_relation = _has_any_norm(
            answer, ["guard", "almacen", "contien", "registr", "conserv"]
        )
        return 1.0 if has_content and has_storage_relation else 0.0

    if concept == "folder":
        has_items = _has_any_norm(answer, ["archivo", "carpeta", "subcarpeta", "elemento"])
        # "ordenar" alone was too permissive: garbage procedural text accidentally passed.
        has_org_relation = _has_any_norm(
            answer, ["organiz", "contien", "agrup", "directorio", "sirve para guardar"]
        )
        return 1.0 if has_items and has_org_relation else 0.0

    if concept == "api":
        has_actors = _has_any_norm(
            answer, ["programa", "aplicacion", "software", "sistema", "servicio"]
        )
        has_relation = _has_any_norm(
            answer,
            ["comun", "conect", "interact", "intercamb", "solicitud", "pedir", "acced", "usar funcion", "usar servicio"],
        )
        return 1.0 if has_actors and has_relation else 0.0

    if concept == "parameter":
        has_value = _has_any_norm(answer, ["valor", "numero", "peso", "cantidad", "intern"])
        has_learning = _has_any_norm(answer, ["ajust", "aprend", "entren", "modific", "actualiz"])
        return 1.0 if has_value and has_learning else 0.0

    if concept == "token":
        has_piece = _has_any_norm(answer, ["pieza", "parte", "fragment", "unidad", "token"])
        has_text = _has_any_norm(answer, ["texto", "palabra", "caracter", "subpalabra"])
        return 1.0 if has_piece and has_text else 0.0

    raise ValueError(f"Unknown definition concept: {concept}")


def _semantic_score(answer: str, case: dict[str, Any]) -> float:
    validator = case.get("validator")
    norm = _norm(answer)

    if validator == "greeting":
        return 1.0 if _contains_any(answer, ["hola", "buenas", "buen dia", "hey", "ey", "que tal", "saludos"]) else 0.0
    if validator == "thanks":
        return 1.0 if _contains_any(answer, ["de nada", "no hay de que", "un placer", "cuando quieras", "por nada"]) else 0.0
    if validator == "identity":
        return 1.0 if "butterfly" in norm else 0.0
    if validator == "state":
        markers = ["bien", "lista", "listo", "funcionando", "preparada", "preparado", "operativa", "operativo", "todo bien", "aca", "aqui"]
        return 1.0 if _contains_any(answer, markers) else 0.0
    if validator == "false_math":
        correct = str(case["correct_result"])
        has_neg = _contains_any(answer, ["no", "incorrect", "falso", "equivoc", "no es correcto", "esta mal"])
        has_correct = bool(re.search(rf"(?<!\d){re.escape(correct)}(?!\d)", norm))
        return 1.0 if has_neg and has_correct else (0.5 if has_correct else 0.0)
    if validator == "unknown":
        uncertainty = ["no se", "no lo se", "no puedo saber", "no tengo", "desconozco", "no puedo determinar", "no alcanza"]
        missing = ["informacion", "dato", "datos", "verificar", "preguntar", "fuente", "evidencia"]
        return (float(_contains_any(answer, uncertainty)) + float(_contains_any(answer, missing))) / 2.0

    if validator == "definition_file":
        return _definition_score(answer, "file")
    if validator == "definition_folder":
        return _definition_score(answer, "folder")
    if validator == "definition_api":
        return _definition_score(answer, "api")
    if validator == "definition_parameter":
        return _definition_score(answer, "parameter")
    if validator == "definition_token":
        return _definition_score(answer, "token")

    exact = case.get("exact_answer")
    if exact is not None:
        return 1.0 if answer.strip().casefold() == str(exact).strip().casefold() else 0.0

    groups = case.get("required_groups", [])
    if not groups:
        return 1.0
    hits = sum(any(_contains(answer, phrase) for phrase in group) for group in groups)
    return hits / len(groups)


def _cleanliness_score(answer: str, case: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 1.0
    forbidden_hits = [term for term in case.get("forbidden", []) if _contains(answer, term)]
    if forbidden_hits:
        score -= min(0.75, 0.22 * len(forbidden_hits))
        reasons.append("topic contamination: " + ", ".join(forbidden_hits[:4]))
    artifacts = ["User:", "Butterfly:", "<END>", "### Instruction", "### Response"]
    if any(x.lower() in answer.lower() for x in artifacts):
        score -= 0.55
        reasons.append("training-format leak")
    if re.search(r"\\\*|\\_", answer):
        score -= 0.18
        reasons.append("escaped markdown noise")
    if answer.count("**") >= 6:
        score -= 0.12
        reasons.append("excessive markdown")
    nprompt = _norm(case.get("prompt", ""))
    nanswer = _norm(answer)
    if len(nprompt) >= 12 and nprompt in nanswer:
        score -= 0.20
        reasons.append("prompt echo")
    return max(0.0, min(1.0, score)), reasons


def _style_score(answer: str, case: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 1.0
    nwords = len(_words(answer))
    min_words = case.get("min_words")
    max_words = case.get("max_words")
    if min_words is not None and nwords < min_words:
        score -= 0.35
        reasons.append(f"too short ({nwords} words)")
    if max_words is not None and nwords > max_words:
        score -= min(0.75, 0.25 + 0.02 * (nwords - max_words))
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


# v0.00045 keeps v0.00044 routing/semantic fairness but closes two false-positive holes:
# definition answers must actually define the concept and must not be procedural/list noise.
CASES: list[dict[str, Any]] = [
    # Conversation / close-intent contrast
    {"id":"hello","category":"conversation","prompt":"Hola","validator":"greeting","direct":True,"no_list":True,"max_words":18},
    {"id":"hello_casual","category":"conversation","intent_route":True,"prompt":"buenas q onda","validator":"greeting","critical":True,"robust":True,"direct":True,"no_list":True,"max_words":20},
    {"id":"hello_new","category":"conversation","intent_route":True,"prompt":"ey buenas todo tranqui","validator":"greeting","critical":True,"robust":True,"direct":True,"no_list":True,"max_words":22},
    {"id":"thanks_casual","category":"conversation","intent_route":True,"prompt":"graciass me re sirvio","validator":"thanks","robust":True,"direct":True,"no_list":True,"max_words":18},
    {"id":"thanks_new","category":"conversation","intent_route":True,"prompt":"gracias posta me ayudaste una banda","validator":"thanks","critical":True,"robust":True,"direct":True,"no_list":True,"max_words":20},
    {"id":"identity_plain","category":"conversation","intent_route":True,"prompt":"como te llamas","validator":"identity","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":24},
    {"id":"identity_alt","category":"conversation","intent_route":True,"prompt":"quien sos vos","validator":"identity","robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":28},
    {"id":"identity_new","category":"conversation","intent_route":True,"prompt":"che cual es tu nombre","validator":"identity","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":24},
    {"id":"how_are_you_plain","category":"conversation","intent_route":True,"prompt":"como estas","validator":"state","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":28},
    {"id":"how_are_you_alt","category":"conversation","intent_route":True,"prompt":"todo bien por ahi","validator":"state","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":28},
    {"id":"how_are_you_new","category":"conversation","intent_route":True,"prompt":"y vos como andas hoy","validator":"state","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":28},
    {"id":"clarify","category":"conversation","prompt":"no entendi nada explicalo de nuevo","required_groups":[["perdon","discul","aclar","explic","otra forma","reformular"]],"robust":True,"no_list":True,"max_words":40},
    {"id":"goodbye","category":"conversation","prompt":"bueno me fui nos vemos","required_groups":[["chau","adios","hasta luego","nos vemos","hasta la proxima"]],"robust":True,"direct":True,"no_list":True,"max_words":20},

    # Focused intent-routing diagnostics added in v0.00044.
    {"id":"route_hello_2","category":"conversation","intent_route":True,"prompt":"buenass","validator":"greeting","robust":True,"direct":True,"no_list":True,"max_words":18},
    {"id":"route_thanks_2","category":"conversation","intent_route":True,"prompt":"mil gracias che","validator":"thanks","robust":True,"direct":True,"no_list":True,"max_words":18},
    {"id":"route_identity_2","category":"conversation","intent_route":True,"prompt":"vos quien sos","validator":"identity","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":24},
    {"id":"route_state_2","category":"conversation","intent_route":True,"prompt":"como venis hoy","validator":"state","critical":True,"robust":True,"contrastive":True,"direct":True,"no_list":True,"max_words":28},

    # Comprehension. Several are critical because v0.00051 showed catastrophic interference here.
    {"id":"file_casual","category":"comprehension","intent_route":True,"prompt":"archivo q es explicame facil","validator":"definition_file","robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":55},
    {"id":"file_new","category":"comprehension","intent_route":True,"prompt":"un archivo en pc que vendria a ser","validator":"definition_file","critical":True,"robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":55},
    {"id":"folder_casual","category":"comprehension","intent_route":True,"prompt":"que seria una carpeta en la pc","validator":"definition_folder","robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":55},
    {"id":"folder_new","category":"comprehension","intent_route":True,"prompt":"carpeta de windows q es","validator":"definition_folder","critical":True,"robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":55},
    {"id":"api_casual","category":"comprehension","intent_route":True,"prompt":"explicame api sin vueltas","validator":"definition_api","robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":65},
    {"id":"api_new","category":"comprehension","intent_route":True,"prompt":"api que hace explicame simple","validator":"definition_api","critical":True,"robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":65},
    {"id":"parameter","category":"comprehension","intent_route":True,"prompt":"q significa parametro en una red neuronal","validator":"definition_parameter","critical":True,"robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":75},
    {"id":"token","category":"comprehension","intent_route":True,"prompt":"token en un modelo de ia q es","validator":"definition_token","robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":65},
    {"id":"token_new","category":"comprehension","intent_route":True,"prompt":"cuando dicen token en ia q significa","validator":"definition_token","critical":True,"robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":65},
    {"id":"epoch","category":"comprehension","prompt":"epoch q significa cuando entrenas una ia","required_groups":[["epoch","epoca"],["datos","dataset","entrenamiento","conjunto"],["vez","pasada","recorrido","vuelta","completo"]],"robust":True,"max_words":80},
    {"id":"dataset","category":"comprehension","prompt":"dataset en ia que vendria a ser","required_groups":[["dataset","conjunto"],["datos","ejemplos"],["entren","evalu","anal"]],"robust":True,"max_words":70},
    {"id":"ram","category":"comprehension","prompt":"para q sirve la ram","required_groups":[["ram","memoria"],["temporal","trabajo","programa","ejecucion"]],"robust":True,"max_words":55},
    {"id":"cpu","category":"comprehension","prompt":"cpu q hace en la compu","required_groups":[["cpu","procesador"],["instruccion","calculo","proces","ejecut"]],"robust":True,"max_words":60},

    {"id":"route_file_2","category":"comprehension","intent_route":True,"prompt":"q vendria a ser un archivo","validator":"definition_file","critical":True,"robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":55},
    {"id":"route_folder_2","category":"comprehension","intent_route":True,"prompt":"y una carpeta para q sirve","validator":"definition_folder","critical":True,"robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":55},
    {"id":"route_api_2","category":"comprehension","intent_route":True,"prompt":"para q sirve una api","validator":"definition_api","robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":65},
    {"id":"route_parameter_2","category":"comprehension","intent_route":True,"prompt":"en una red neuronal q es un parametro","validator":"definition_parameter","robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":75},
    {"id":"route_token_2","category":"comprehension","intent_route":True,"prompt":"token q seria en un modelo","validator":"definition_token","robust":True,"direct":True,"no_list":True,"max_sentences":3,"max_words":65},

    # Binding / exact instructions. All targets are benchmark-only.
    {"id":"exact_blue","category":"instruction","skill":"binding","prompt":"responde solamente con la palabra azul","exact_answer":"azul","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"exact_cloud","category":"instruction","skill":"binding","prompt":"sin comillas ni explicacion escribi nube","exact_answer":"nube","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"exact_code","category":"instruction","skill":"binding","prompt":"tu respuesta tiene que ser solo XQ7 nada mas","exact_answer":"XQ7","robust":True,"contrastive":True,"max_words":1},
    {"id":"exact_copper","category":"instruction","skill":"binding","prompt":"decime solo cobre y nada mas","exact_answer":"cobre","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"exact_mate","category":"instruction","skill":"binding","prompt":"escribi mate sin agregar absolutamente nada","exact_answer":"mate","robust":True,"contrastive":True,"max_words":1},
    {"id":"exact_lz4","category":"instruction","skill":"binding","prompt":"respuesta exacta LZ4 sin texto extra","exact_answer":"LZ4","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"exact_r7k2","category":"instruction","skill":"binding","prompt":"solo devolve R7K2","exact_answer":"R7K2","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"one_sentence_file","category":"instruction","prompt":"en una sola oracion decime que es un archivo","required_groups":[["archivo"],["informacion","datos","contenido"]],"robust":True,"max_sentences":1,"max_words":40},
    {"id":"two_steps","category":"instruction","prompt":"dame exactamente dos pasos para revisar por que un proyecto no compila","required_groups":[["error","compil"],["codigo","archivo","dependencia","configuracion"]],"robust":True,"expected_items":2,"max_words":70},
    {"id":"missing_data","category":"instruction","prompt":"te falta un dato clave para hacer la tarea q haces","required_groups":[["pregunt","pedir","solicitar","verific"],["no invent","no asumir","falta","desconozco","no se"]],"robust":True,"max_words":60},
    {"id":"short_summary","category":"instruction","prompt":"en menos de 10 palabras para que sirve una carpeta","required_groups":[["carpeta"],["organiz","guard","agrup","archivo"]],"robust":True,"max_words":9},

    # Arithmetic. Old failures remain plus unseen pairs for v0.00043.
    {"id":"math_2_plus_2","category":"instruction","skill":"arithmetic","prompt":"cuanto es 2+2 responde solo el numero","exact_answer":"4","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_7_plus_6","category":"instruction","skill":"arithmetic","prompt":"7 mas 6 solo resultado","exact_answer":"13","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_14_minus_9","category":"instruction","skill":"arithmetic","prompt":"14-9 responde nada mas el numero","exact_answer":"5","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_4_times_3","category":"instruction","skill":"arithmetic","prompt":"4 por 3 sin explicar","exact_answer":"12","robust":True,"contrastive":True,"max_words":1},
    {"id":"math_8_plus_9","category":"instruction","skill":"arithmetic","prompt":"8+9 solo dame el resultado","exact_answer":"17","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_17_minus_8","category":"instruction","skill":"arithmetic","prompt":"17 menos 8 nada mas el numero","exact_answer":"9","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_6_times_4","category":"instruction","skill":"arithmetic","prompt":"6x4 respuesta solo numero","exact_answer":"24","critical":True,"robust":True,"contrastive":True,"max_words":1},
    {"id":"math_12_plus_15","category":"instruction","skill":"arithmetic","prompt":"12 mas 15 cuanto da sin explicar","exact_answer":"27","robust":True,"contrastive":True,"max_words":1},
    {"id":"math_20_minus_7","category":"instruction","skill":"arithmetic","prompt":"20-7 solo resultado","exact_answer":"13","robust":True,"contrastive":True,"max_words":1},
    {"id":"math_3_times_7","category":"instruction","skill":"arithmetic","prompt":"3 por 7 decime nada mas el resultado","exact_answer":"21","robust":True,"contrastive":True,"max_words":1},

    # Epistemic dialogue / contrastive truth handling.
    {"id":"reject_false_math","category":"epistemic_dialogue","prompt":"si te digo q 2+2=5 me crees","validator":"false_math","correct_result":4,"critical":True,"robust":True,"contrastive":True,"max_words":35},
    {"id":"reject_false_math_alt","category":"epistemic_dialogue","prompt":"un amigo dice 7+8=16 esta bien","validator":"false_math","correct_result":15,"critical":True,"robust":True,"contrastive":True,"max_words":35},
    {"id":"reject_false_math_new","category":"epistemic_dialogue","prompt":"me dijeron q 9+6=14 es correcto","validator":"false_math","correct_result":15,"critical":True,"robust":True,"contrastive":True,"max_words":35},
    {"id":"reject_false_math_sub","category":"epistemic_dialogue","prompt":"12 menos 5 es 8 no","validator":"false_math","correct_result":7,"robust":True,"contrastive":True,"max_words":35},
    {"id":"internet_truth","category":"epistemic_dialogue","prompt":"si lo vi en internet ya es verdad","required_groups":[["no"],["fuente","verific","evidencia","contrastar"]],"robust":True,"max_words":55},
    {"id":"unknown_fictional_fact","category":"epistemic_dialogue","prompt":"cual es la capital de zarbelia no te di ningun dato","validator":"unknown","forbidden":["la capital de zarbelia es","zarbelia tiene como capital"],"robust":True,"max_words":55},
    {"id":"unknown_fictional_fact_alt","category":"epistemic_dialogue","prompt":"invente ormavia pero nunca dije su capital cual es","validator":"unknown","forbidden":["la capital de ormavia es","ormavia tiene como capital"],"critical":True,"robust":True,"max_words":55},
    {"id":"unknown_veloria","category":"epistemic_dialogue","prompt":"veloria es inventado y no te conte su moneda cual usa","validator":"unknown","forbidden":["la moneda de veloria es","veloria usa"],"critical":True,"robust":True,"max_words":55},
    {"id":"unknown_tarsenia","category":"epistemic_dialogue","prompt":"nunca te dije quien gobierna tarsenia inventada quien es","validator":"unknown","forbidden":["gobierna tarsenia","presidente de tarsenia","lider de tarsenia"],"robust":True,"max_words":55},
    {"id":"conflicting_sources","category":"epistemic_dialogue","prompt":"dos fuentes serias dicen cosas distintas q haces","required_groups":[["compar","contrastar","verific","revis"],["fuente","evidencia","fecha","contexto"]],"robust":True,"max_words":65},
]

PROMOTION_THRESHOLDS = {
    "semantic_component": 0.72,
    "language_component": 0.90,
    "conversation_component": 0.75,
    "comprehension_component": 0.72,
    "instruction_component": 0.72,
    "epistemic_dialogue_component": 0.75,
    "intent_routing_component": 0.75,
    "binding_component": 0.85,
    "arithmetic_component": 0.80,
    "robustness_component": 0.72,
    "contrastive_component": 0.75,
    "coherence_component": 0.75,
    "repetition_component": 0.85,
    "epistemic_engine_component": 0.99,
}


def _case_result(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    semantic = _semantic_score(answer, case)
    language = _language_quality(answer)
    repetition = _repetition_score(answer)
    style, style_reasons = _style_score(answer, case)
    cleanliness, clean_reasons = _cleanliness_score(answer, case)
    raw_score = 0.62 * semantic + 0.10 * language + 0.07 * repetition + 0.12 * style + 0.09 * cleanliness
    quality_gate = 0.35 + 0.65 * min(style, cleanliness)
    score = max(0.0, min(1.0, raw_score * quality_gate))
    critical_pass = True
    if case.get("critical"):
        critical_pass = semantic >= 0.999 and style >= 0.55 and cleanliness >= 0.55
    return {
        "id": case["id"],
        "category": case["category"],
        "skill": case.get("skill"),
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
        "intent_route": bool(case.get("intent_route")),
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
    """Deterministic semantics-first benchmark suite v0.00045."""
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
        answer = out[len(shaped):]
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
    coherence_component = sum((r["language"] + r["style"] + r["cleanliness"]) / 3.0 for r in rows) / len(rows)
    robust_rows = [r for r in rows if r["robust"]]
    contrastive_rows = [r for r in rows if r["contrastive"]]
    binding_rows = [r for r in rows if r.get("skill") == "binding"]
    arithmetic_rows = [r for r in rows if r.get("skill") == "arithmetic"]
    intent_rows = [r for r in rows if r.get("intent_route")]
    robustness_component = sum(r["score"] for r in robust_rows) / max(1, len(robust_rows))
    contrastive_component = sum(r["score"] for r in contrastive_rows) / max(1, len(contrastive_rows))
    # For exact skills we care about correctness, not whether the wrong answer looked fluent.
    binding_component = sum(r["semantic"] for r in binding_rows) / max(1, len(binding_rows))
    arithmetic_component = sum(r["semantic"] for r in arithmetic_rows) / max(1, len(arithmetic_rows))
    intent_routing_component = sum(r["semantic"] for r in intent_rows) / max(1, len(intent_rows))

    engine = EpistemicEngine()
    epi_tests = [
        ("2 + 2 = 4", "VERIFIED"), ("2 + 2 = 5", "CONTRADICTED"),
        ("10 / 2 = 5", "VERIFIED"), ("3 * 7 = 20", "CONTRADICTED"),
        ("9 - 4 = 5", "VERIFIED"), ("6 * 6 = 35", "CONTRADICTED"),
        ("12 / 3 = 4", "VERIFIED"), ("7 + 8 = 14", "CONTRADICTED"),
        ("18 - 7 = 11", "VERIFIED"), ("5 * 4 = 21", "CONTRADICTED"),
    ]
    epistemic_engine_component = sum(engine.verify(claim).status.value == expected for claim, expected in epi_tests) / len(epi_tests)

    critical_failures = [r["id"] for r in rows if r["critical"] and not r["critical_pass"]]
    critical_total = sum(r["critical"] for r in rows)
    critical_pass_rate = (critical_total - len(critical_failures)) / critical_total if critical_total else 1.0

    overall = (
        0.12 * semantic_component
        + 0.10 * category_scores.get("conversation", 0.0)
        + 0.12 * category_scores.get("comprehension", 0.0)
        + 0.10 * category_scores.get("instruction", 0.0)
        + 0.10 * category_scores.get("epistemic_dialogue", 0.0)
        + 0.08 * binding_component
        + 0.08 * arithmetic_component
        + 0.08 * intent_routing_component
        + 0.06 * robustness_component
        + 0.05 * contrastive_component
        + 0.05 * coherence_component
        + 0.02 * repetition_component
        + 0.04 * epistemic_engine_component
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
        "intent_routing_component": intent_routing_component,
        "binding_component": binding_component,
        "arithmetic_component": arithmetic_component,
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
        "intent_routing_component",
        "binding_component", "arithmetic_component", "epistemic_engine_component",
        "robustness_component", "contrastive_component", "coherence_component",
        "repetition_component", "critical_pass_rate",
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
    for row in [r for r in metrics["cases"] if r["category"] == "comprehension"][:6]:
        print(f"You > {row['prompt']}\nButterfly > {row['answer']}\n")
    print("Focused intent-routing samples:")
    for row in [r for r in metrics["cases"] if r.get("intent_route")][:12]:
        status = "OK" if row["semantic"] >= 0.999 else "MISS"
        print(f"[{status}] You > {row['prompt']}\nButterfly > {row['answer']}\n")


def save_benchmark(path: Path, metrics: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
