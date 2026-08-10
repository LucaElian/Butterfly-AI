from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import random
import re
import unicodedata

from ..config import CORPUS_DIR, ROOT, save_json
from ..learning.evaluator import (
    CASES,
    BENCHMARK_RESERVED_EXACT_TARGETS,
    BENCHMARK_RESERVED_MATH,
    BENCHMARK_RESERVED_FALSE_MATH,
    BENCHMARK_RESERVED_FICTIONAL,
)

V51_DIR = CORPUS_DIR / "v00051"
MANIFEST = V51_DIR / "manifest.json"

STAGE_FILES = {
    "robust_dialogue": (V51_DIR / "robust_dialogue_train.jsonl", V51_DIR / "robust_dialogue_valid.jsonl"),
    "binding_math": (V51_DIR / "binding_math_train.jsonl", V51_DIR / "binding_math_valid.jsonl"),
    "epistemic_contrast": (V51_DIR / "epistemic_contrast_train.jsonl", V51_DIR / "epistemic_contrast_valid.jsonl"),
    "mixed_generalization": (V51_DIR / "mixed_generalization_train.jsonl", V51_DIR / "mixed_generalization_valid.jsonl"),
}


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _norm(text: str) -> str:
    """Leak-resistant surface normalization.

    Accent marks, punctuation, case and the casual q/que spelling do not make a
    benchmark sentence 'new'. This prevents us from accidentally training on
    essentially the same prompt with only cosmetic changes.
    """
    text = _strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9+\-*/ ]+", " ", text)
    words = re.sub(r"\s+", " ", text).strip().split()
    words = ["que" if w == "q" else "porque" if w == "xq" else w for w in words]
    return " ".join(words)


def _raw_norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _casual_surface_variants(row: dict) -> list[dict]:
    """Create raw-input variants without pretending they are new semantics.

    The benchmark leak check still uses _norm(), so punctuation/accent changes
    can NEVER turn a benchmark prompt into training data. Inside training, though,
    Butterfly is allowed to see the same intent written by a human in messier ways.
    """
    text = row["user"]
    variants = [("raw", text)]
    no_punct = re.sub(r"[¿?¡!.,;:\"']+", "", text)
    variants.append(("nopunct", no_punct))
    variants.append(("noaccent", _strip_accents(text)))
    q_form = re.sub(r"\bque\b", "q", _strip_accents(no_punct), flags=re.IGNORECASE)
    variants.append(("casual_q", q_form))
    out = []
    seen = set()
    for tag, value in variants:
        key = _raw_norm(value)
        if not key or key in seen:
            continue
        seen.add(key)
        copy = dict(row)
        copy["user"] = value.strip()
        copy["surface_family"] = f"{row.get('surface_family','train')}_{tag}"
        out.append(copy)
    return out


def _augment_train_surfaces(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.extend(_casual_surface_variants(row))
    return _dedupe(out)


BENCHMARK_PROMPTS = {_norm(c["prompt"]) for c in CASES}


def _row(user: str, assistant: str, category: str, surface_family: str) -> dict:
    return {
        "user": user.strip(),
        "assistant": assistant.strip(),
        "category": category,
        "surface_family": surface_family,
    }


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        raw_key = _raw_norm(row["user"])
        leak_key = _norm(row["user"])
        if not raw_key or raw_key in seen:
            continue
        if leak_key in BENCHMARK_PROMPTS:
            continue
        seen.add(raw_key)
        out.append(row)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _repeat_balanced(rows: list[dict], target_rows: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)
    if not by_cat:
        raise RuntimeError("Cannot balance an empty alignment dataset")
    out = [dict(row) for row in rows]
    cats = sorted(by_cat)
    while len(out) < target_rows:
        for cat in cats:
            if len(out) >= target_rows:
                break
            out.append(dict(rng.choice(by_cat[cat])))
    rng.shuffle(out)
    return out


def _social_and_definitions() -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    valid: list[dict] = []

    greeting_train = [
        "holaa", "buen dia butterfly", "buenas tardes", "hey butterfly", "ey estas ahi",
        "saludos", "hola de nuevo", "buenasss", "hola todo tranqui", "hola estas por ahi",
        "que tal butterfly", "ey hola", "muy buenas", "holi", "hola hola",
    ]
    greeting_valid = ["holi butterfly", "buenass todo tranqui", "hey estas ahi", "saludoss", "muy buen dia"]
    greeting_answers = [
        "¡Hola! ¿En qué te puedo ayudar?", "¡Buenas! ¿Qué necesitás?",
        "¡Hey! Decime qué querés hacer.", "Hola. Estoy lista para ayudarte.",
    ]
    for i, p in enumerate(greeting_train):
        train.append(_row(p, greeting_answers[i % len(greeting_answers)], "greeting", "greeting_train_a"))
    for i, p in enumerate(greeting_valid):
        valid.append(_row(p, greeting_answers[(i + 1) % len(greeting_answers)], "greeting", "greeting_valid_z"))

    thanks_train = [
        "muchas gracias", "mil gracias", "te agradezco", "gracias posta", "genial gracias por la ayuda",
        "gracias me sirvio", "buenisimo gracias", "joya gracias", "gracias por explicarlo", "gracias che",
    ]
    thanks_valid = ["te re agradezco", "gracias estuvo clarisimo", "mil gracias posta", "joya te agradezco"]
    thanks_answers = ["De nada.", "¡De nada!", "Un placer.", "Cuando quieras."]
    for i, p in enumerate(thanks_train):
        train.append(_row(p, thanks_answers[i % len(thanks_answers)], "thanks", "thanks_train_a"))
    for i, p in enumerate(thanks_valid):
        valid.append(_row(p, thanks_answers[(i + 1) % len(thanks_answers)], "thanks", "thanks_valid_z"))

    # Contrastive intents deliberately share words such as "como" and "estas".
    identity_train = [
        "cual es tu nombre", "decime tu nombre", "quien eres", "sos butterfly", "con quien estoy hablando",
        "presentate cortito", "nombre de esta ia", "como se llama esta inteligencia artificial",
        "decime quien sos", "vos sos butterflyai",
    ]
    identity_valid = ["con quien hablo ahora", "decime como te dicen", "cual seria tu nombre", "quien es la ia que responde"]
    identity_answers = ["Me llamo ButterflyAI.", "Soy ButterflyAI.", "Soy ButterflyAI, una inteligencia artificial local en desarrollo."]
    for i, p in enumerate(identity_train):
        train.append(_row(p, identity_answers[i % len(identity_answers)], "identity", "identity_train_a"))
    for i, p in enumerate(identity_valid):
        valid.append(_row(p, identity_answers[(i + 1) % len(identity_answers)], "identity", "identity_valid_z"))

    state_train = [
        "todo bien", "como andas hoy", "estas funcionando", "estas lista", "andas bien", "todo en orden",
        "estas preparada", "como venis", "funcionas bien", "estas ok",
    ]
    state_valid = ["andas todo bien", "estas operativa", "como venis hoy", "todo tranquilo por ahi"]
    state_answers = [
        "Bien, gracias. Estoy lista para ayudarte.", "Sí, estoy funcionando y lista para ayudarte.",
        "Sí. Estoy preparada para ayudarte.",
    ]
    for i, p in enumerate(state_train):
        train.append(_row(p, state_answers[i % len(state_answers)], "state", "state_train_a"))
    for i, p in enumerate(state_valid):
        valid.append(_row(p, state_answers[(i + 1) % len(state_answers)], "state", "state_valid_z"))

    clarify_train = ["no entendi", "me perdi", "eso quedo raro", "explicalo otra vez", "no te segui", "reformula eso", "decilo mas simple"]
    clarify_valid = ["no capte lo que quisiste decir", "eso no se entendio", "volvelo a explicar mas claro"]
    clarify_answer = "Perdón. Lo explico de otra forma y más directo."
    for p in clarify_train:
        train.append(_row(p, clarify_answer, "clarify", "clarify_train_a"))
    for p in clarify_valid:
        valid.append(_row(p, clarify_answer, "clarify", "clarify_valid_z"))

    definitions = {
        "archivo": "Un archivo es una unidad de información guardada que puede contener texto, imágenes, datos u otro contenido.",
        "carpeta": "Una carpeta es un contenedor usado para organizar archivos y otras carpetas.",
        "API": "Una API es una interfaz que permite que programas o sistemas se comuniquen mediante reglas y operaciones definidas.",
        "token": "Un token es una unidad de texto que un modelo procesa, como una palabra, parte de una palabra o signo.",
        "epoch": "Un epoch es una pasada completa por el conjunto de datos de entrenamiento.",
        "parámetro de una red neuronal": "Un parámetro es un valor numérico ajustable que la red aprende durante el entrenamiento.",
        "RAM": "La RAM es memoria de trabajo rápida y temporal usada por los programas mientras están en ejecución.",
        "CPU": "La CPU es el procesador principal que ejecuta instrucciones y realiza cálculos generales.",
        "GPU": "Una GPU es un procesador especializado en muchas operaciones paralelas.",
        "batch": "Un batch es un grupo de ejemplos procesados juntos antes de actualizar los parámetros.",
        "checkpoint": "Un checkpoint es un estado guardado de los pesos de un modelo.",
        "dataset": "Un dataset es un conjunto organizado de datos usado para análisis, evaluación o entrenamiento.",
        "variable": "Una variable es un nombre asociado a un valor que un programa puede consultar o modificar.",
        "función": "Una función es un bloque reutilizable de instrucciones que puede recibir datos y producir un resultado.",
    }
    train_templates = [
        "definime {t} facil", "que significa {t}", "explicame {t} corto", "{t} que seria",
        "contame simple que es {t}", "en informatica que quiere decir {t}",
    ]
    valid_templates = [
        "necesito entender {t} en pocas palabras", "como le explicarias {t} a alguien nuevo", "a que se refiere {t} exactamente",
    ]
    for term, answer in definitions.items():
        for j, tmpl in enumerate(train_templates):
            train.append(_row(tmpl.format(t=term), answer, "definition", f"definition_train_{j}"))
        for j, tmpl in enumerate(valid_templates):
            valid.append(_row(tmpl.format(t=term), answer, "definition", f"definition_valid_{j}"))

    # Small stable QA teaches direct relevance in ordinary Spanish.
    qa = [
        ("cuantos dias tiene una semana", "Siete."), ("cuantos meses tiene un año", "Doce."),
        ("que animal dice miau", "El gato."), ("que viene despues del lunes", "Martes."),
        ("cual es el primer mes del año", "Enero."), ("que es mayor 10 o 7", "10."),
        ("cuanto es 5 por 5", "25."), ("12 dividido 3 cuanto da", "4."),
    ]
    for q, a in qa:
        train.append(_row(q, a, "short_qa", "qa_train_a"))
        valid.append(_row("respondeme directo " + q, a, "short_qa", "qa_valid_z"))

    return _augment_train_surfaces(_dedupe(train)), _dedupe(valid)


def _binding_and_math() -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    valid: list[dict] = []

    # Exact-copy values: benchmark targets are not present in either train or valid.
    train_targets = [
        "rojo","verde","amarillo","violeta","naranja","negro","blanco","gris","marron","cian",
        "gato","perro","caballo","zorro","pato","pez","casa","mesa","silla","puerta","ventana",
        "sol","luna","agua","fuego","arbol","libro","papel","vaso","cafe","mate","pan","llave",
        "norte","sur","este","oeste","enero","marzo","martes","jueves","sabado","rapido","simple",
        "claro","listo","hecho","correcto","uno","dos","tres","cinco","seis","ocho","nueve",
        "AB7","KM2","RT9","LP4","ZC8","M3N","Q5P","R8A","T2K","V6B",
    ]
    valid_targets = [
        "celeste","dorado","conejo","rio","montaña","techo","domingo","abril","lento","breve",
        "JD6","PK3","N7R","C4M","F9T",
    ]
    reserved = {x.casefold() for x in BENCHMARK_RESERVED_EXACT_TARGETS}
    assert not ({x.casefold() for x in train_targets + valid_targets} & reserved)

    train_templates = [
        "responde solo {x}", "decime unicamente {x}", "pone {x} y nada mas", "solo quiero {x}",
        "tu respuesta tiene que ser {x}", "sin explicar escribi {x}", "contestame con {x} solamente",
        "devolve nada mas {x}", "escribi exactamente {x}",
    ]
    valid_templates = [
        "la respuesta debe contener solo {x}", "no agregues nada responde {x}", "escribi nada mas que {x}",
        "quiero como unica salida {x}",
    ]
    for target in train_targets:
        for j, tmpl in enumerate(train_templates):
            train.append(_row(tmpl.format(x=target), target, "exact_copy", f"copy_train_{j}"))
    for target in valid_targets:
        for j, tmpl in enumerate(valid_templates):
            valid.append(_row(tmpl.format(x=target), target, "exact_copy", f"copy_valid_{j}"))

    train_math_templates = [
        "calcula {a}{op}{b} responde solo el numero", "{a} {word} {b} solo resultado",
        "sin explicar cuanto da {a}{op}{b}", "resolve {a}{op}{b} y pone nada mas el resultado",
    ]
    valid_math_templates = [
        "quiero unicamente el resultado de {a}{op}{b}", "hace {a} {word} {b} sin texto extra",
    ]
    word = {"+":"mas", "-":"menos", "*":"por"}

    pairs: list[tuple[str,int,int,int]] = []
    for a in range(0, 21):
        for b in range(0, 21):
            pairs.append(("+", a, b, a + b))
    for a in range(1, 31):
        for b in range(0, a + 1):
            pairs.append(("-", a, b, a - b))
    for a in range(0, 13):
        for b in range(0, 13):
            pairs.append(("*", a, b, a * b))

    for op, a, b, result in pairs:
        if (op, a, b) in BENCHMARK_RESERVED_MATH:
            continue
        # Operand-pair split, not random prompt split. Validation combinations are
        # never shown during training even under a different wording.
        held_out = ((a * 37 + b * 17 + ord(op)) % 7 == 0)
        templates = valid_math_templates if held_out else train_math_templates
        dest = valid if held_out else train
        prefix = "math_valid" if held_out else "math_train"
        for j, tmpl in enumerate(templates):
            dest.append(_row(tmpl.format(a=a, b=b, op=op, word=word[op]), str(result), "arithmetic", f"{prefix}_{j}"))

    return _augment_train_surfaces(_dedupe(train)), _dedupe(valid)


def _epistemic_and_contrast() -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    valid: list[dict] = []

    # Minimal contrastive pairs: very similar surface, different intent/answer.
    contrast_train = [
        ("decime tu nombre", "Me llamo ButterflyAI.", "contrast_identity"),
        ("decime como estas", "Bien, gracias. Estoy lista para ayudarte.", "contrast_state"),
        ("quien sos", "Soy ButterflyAI.", "contrast_identity"),
        ("como andas", "Bien, gracias. Estoy lista para ayudarte.", "contrast_state"),
        ("estas lista", "Sí. Estoy preparada para ayudarte.", "contrast_state"),
        ("sos butterfly", "Sí. Soy ButterflyAI.", "contrast_identity"),
    ]
    contrast_valid = [
        ("quien es la ia que me responde", "Soy ButterflyAI.", "contrast_identity"),
        ("andas bien hoy", "Bien, gracias. Estoy lista para ayudarte.", "contrast_state"),
        ("cual es el nombre de esta ia", "Me llamo ButterflyAI.", "contrast_identity"),
        ("estas funcionando bien", "Sí, estoy funcionando y lista para ayudarte.", "contrast_state"),
    ]
    for p, a, c in contrast_train:
        train.append(_row(p, a, c, "contrast_train_a"))
    for p, a, c in contrast_valid:
        valid.append(_row(p, a, c, "contrast_valid_z"))

    false_train_templates = [
        "alguien dice que {a}{op}{b}={wrong} esta bien", "me dijeron {a} {word} {b} da {wrong} verifica",
        "es verdad que {a}{op}{b} es {wrong}",
    ]
    false_valid_templates = [
        "un amigo insiste con que {a}{op}{b}={wrong} tiene razon", "chequear {a} {word} {b} = {wrong}",
    ]
    word = {"+":"mas", "-":"menos", "*":"por"}
    candidates: list[tuple[str,int,int,int,int]] = []
    for op in ("+", "-", "*"):
        for a in range(1, 16):
            for b in range(1, 11):
                if op == "-" and b > a:
                    continue
                true = a + b if op == "+" else a - b if op == "-" else a * b
                wrong = true + (1 if (a + b) % 2 == 0 else 2)
                if (op, a, b, wrong) in BENCHMARK_RESERVED_FALSE_MATH:
                    continue
                candidates.append((op, a, b, true, wrong))
    for op, a, b, true, wrong in candidates:
        held_out = ((a * 29 + b * 11 + ord(op)) % 8 == 0)
        templates = false_valid_templates if held_out else false_train_templates
        dest = valid if held_out else train
        prefix = "false_valid" if held_out else "false_train"
        answer = f"No. {a} {op} {b} = {true}."
        for j, tmpl in enumerate(templates):
            dest.append(_row(tmpl.format(a=a,b=b,op=op,wrong=wrong,word=word[op]), answer, "reject_false_math", f"{prefix}_{j}"))

    left = ["Al","Bel","Cor","Dor","El","Fal","Gar","Lum","Mor","Ner","Pra","Quel","Rin","Tor","Val","Zen"]
    right = ["dania","loria","vania","mora","taria","noria","silia","bria","doria","quenia"]
    names = [a+b for a in left for b in right if (a+b).lower() not in BENCHMARK_RESERVED_FICTIONAL]
    train_unknown_templates = [
        "invente {name} pero no dije su capital cual es", "sin datos decime la capital del pais ficticio {name}",
        "{name} es inventado y nunca defini su capital podes saberla",
    ]
    valid_unknown_templates = [
        "cree {name} recien y no te conte su capital la sabes", "no hay informacion de {name} cual seria su capital",
    ]
    unknown_answers = [
        "No tengo información suficiente para saberlo; necesitaría que me des ese dato.",
        "No puedo determinarlo sin información adicional y no debería inventarlo.",
        "No lo sé con los datos disponibles. Tendría que verificarlo o preguntarte.",
    ]
    for i, name in enumerate(names):
        held_out = (i % 7 == 0)
        templates = valid_unknown_templates if held_out else train_unknown_templates
        dest = valid if held_out else train
        prefix = "unknown_valid" if held_out else "unknown_train"
        for j, tmpl in enumerate(templates):
            dest.append(_row(tmpl.format(name=name), unknown_answers[(i+j)%len(unknown_answers)], "unknown_fact", f"{prefix}_{j}"))

    missing_train = [
        ("quiero abrir algo pero no te dije que archivo", "Me falta saber qué archivo querés abrir. Te lo preguntaría antes de actuar."),
        ("abri una carpeta pero no pase la ruta", "Me falta la ruta o una forma de identificar la carpeta; no debería inventarla."),
        ("hace una reserva mañana pero no dije hora", "Falta el horario necesario. Te lo pediría antes de completar la reserva."),
        ("calcula el total pero no te di importes", "No puedo calcularlo sin los importes; necesito esos datos."),
        ("decime una medicion que nunca te mostre", "No tengo ese dato y no debería inventarlo."),
    ]
    missing_valid = [
        ("te pido mandar un mail pero no dije a quien", "Falta el destinatario. Te lo preguntaría antes de enviar nada."),
        ("quiero copiar un archivo pero no dije destino", "Falta el destino. Necesito ese dato antes de realizar la acción."),
    ]
    for p,a in missing_train:
        train.append(_row(p,a,"missing_information","missing_train_a"))
    for p,a in missing_valid:
        valid.append(_row(p,a,"missing_information","missing_valid_z"))

    source_train = [
        "dos fuentes confiables no coinciden que haces", "una fuente nueva contradice una vieja como lo revisas",
        "dos documentos serios dicen cosas distintas", "si una web afirma algo ya lo das por verdad",
    ]
    source_valid = ["hay dos fuentes buenas con datos incompatibles como decidis", "internet dice algo alcanza para creerlo"]
    for p in source_train:
        ans = "Compararía las fuentes, su fecha, contexto y evidencia antes de decidir qué afirmación está mejor respaldada." if "dos" in p or "fuente" in p else "No. Hay que revisar la fuente, la evidencia, la fecha y contrastar cuando sea necesario."
        train.append(_row(p, ans, "source_verification", "source_train_a"))
    for p in source_valid:
        ans = "Compararía las fuentes, su fecha, contexto y evidencia antes de decidir qué afirmación está mejor respaldada." if "dos" in p else "No. Hay que revisar la fuente, la evidencia, la fecha y contrastar cuando sea necesario."
        valid.append(_row(p, ans, "source_verification", "source_valid_z"))

    return _augment_train_surfaces(_dedupe(train)), _dedupe(valid)


def _assert_clean_split(stage: str, train: list[dict], valid: list[dict]) -> None:
    train_norm = {_norm(r["user"]) for r in train}
    valid_norm = {_norm(r["user"]) for r in valid}
    overlap = train_norm & valid_norm
    if overlap:
        raise RuntimeError(f"v0.00051 train/valid prompt leak in {stage}: {len(overlap)}")
    benchmark_leaks = (train_norm | valid_norm) & BENCHMARK_PROMPTS
    if benchmark_leaks:
        raise RuntimeError(f"v0.00051 benchmark surface leak in {stage}: {sorted(benchmark_leaks)[:3]}")
    train_families = {r.get("surface_family") for r in train}
    valid_families = {r.get("surface_family") for r in valid}
    family_overlap = train_families & valid_families
    if family_overlap:
        raise RuntimeError(f"v0.00051 surface-template family leak in {stage}: {sorted(family_overlap)[:3]}")


def _stage_stats(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "unique_prompts": len({_raw_norm(r["user"]) for r in rows}),
        "normalized_intents": len({_norm(r["user"]) for r in rows}),
        "surface_families": len({r.get("surface_family") for r in rows}),
        "categories": dict(sorted(Counter(r["category"] for r in rows).items())),
        "bytes": sum(len(json.dumps(r, ensure_ascii=False).encode("utf-8")) + 1 for r in rows),
    }


def build_alignment_corpus_v00051() -> dict:
    V51_DIR.mkdir(parents=True, exist_ok=True)

    social_train, social_valid = _social_and_definitions()
    bind_train, bind_valid = _binding_and_math()
    epi_train, epi_valid = _epistemic_and_contrast()

    robust_train = _repeat_balanced(social_train, max(4_000, len(social_train) * 10), 5101)
    binding_train = _repeat_balanced(bind_train, max(7_000, len(bind_train) * 3), 5102)
    epistemic_train = _repeat_balanced(epi_train, max(5_500, len(epi_train) * 4), 5103)

    # Mixed stage keeps all UNIQUE training families and separately all held-out
    # validation families. Validation is not a random slice of train wording.
    mixed_train_unique = _dedupe(social_train + bind_train + epi_train)
    mixed_valid = _dedupe(social_valid + bind_valid + epi_valid)
    mixed_train = _repeat_balanced(mixed_train_unique, max(8_000, len(mixed_train_unique) * 2), 5104)

    datasets = {
        "robust_dialogue": (robust_train, social_valid),
        "binding_math": (binding_train, bind_valid),
        "epistemic_contrast": (epistemic_train, epi_valid),
        "mixed_generalization": (mixed_train, mixed_valid),
    }

    for stage, (train, valid) in datasets.items():
        _assert_clean_split(stage, train, valid)
        train_path, valid_path = STAGE_FILES[stage]
        _write_jsonl(train_path, train)
        _write_jsonl(valid_path, valid)

    manifest = {
        "version": "0.00051",
        "format": "assistant-only-supervised-jsonl-v2",
        "benchmark_suite_held_out": "0.00042",
        "benchmark_surface_leaks": 0,
        "validation_surface_family_overlap": 0,
        "seed_brain": "0.0004",
        "strategy": [
            "Continue from accepted v0.0004 weights; v0.0005 rejected weights are not reused.",
            "Teach robust Spanish input: missing accents/punctuation, casual q/que, elongated greetings and colloquial wording.",
            "Use contrastive intent pairs so identity and state cannot be solved by the word 'como' alone.",
            "Teach variable binding/copy with many values while reserving benchmark targets entirely.",
            "Split arithmetic by operand combinations, not by random prompt rows.",
            "Use held-out validation wording families that never occur in training.",
            "Mask USER targets; loss is paid only on Butterfly response tokens.",
        ],
        "stages": {
            stage: {"train": _stage_stats(train), "valid": _stage_stats(valid)}
            for stage, (train, valid) in datasets.items()
        },
    }
    save_json(MANIFEST, manifest)

    print("ButterflyAI v0.00051 corrective corpus built.")
    print("Benchmark suite held out      : v0.00042")
    print("Benchmark surface leaks       : 0")
    print("Train/valid surface-family overlap: 0")
    print("Casual Spanish included       : no accents / little punctuation / q=que / colloquial variants")
    print("Exact-copy benchmark targets  : RESERVED, never trained")
    print("Arithmetic benchmark pairs    : RESERVED, never trained")
    for stage, info in manifest["stages"].items():
        tr, va = info["train"], info["valid"]
        print(
            f"{stage:24s} train {tr['rows']:,} rows ({tr['unique_prompts']:,} raw unique / {tr['normalized_intents']:,} normalized intents, {tr['surface_families']} surface families) | "
            f"valid {va['rows']:,} unique ({va['surface_families']} held-out families) | {tr['bytes']/1024/1024:.2f} MB train"
        )
    print(f"Manifest: {MANIFEST}")
    return manifest
