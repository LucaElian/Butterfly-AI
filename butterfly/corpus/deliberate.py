from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from ..config import ROOT
from ..learning.evaluator import (
    BENCHMARK_RESERVED_EXACT_TARGETS,
    BENCHMARK_RESERVED_FALSE_MATH,
    BENCHMARK_RESERVED_FICTIONAL,
    BENCHMARK_RESERVED_MATH,
    BENCHMARK_SUITE_VERSION,
    benchmark_surface_prompts,
    normalize_surface,
)

DATA_ROOT = ROOT / "data" / "corpus" / "deliberate"
ARCHIVE_DIR = DATA_ROOT / "archive"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
HISTORY_PATH = DATA_ROOT / "history.json"
SEED = 530043

STAGE_FILES = {
    "robust_comprehension": (DATA_ROOT / "robust_comprehension_train.jsonl", DATA_ROOT / "robust_comprehension_valid.jsonl"),
    "copy_binding": (DATA_ROOT / "copy_binding_train.jsonl", DATA_ROOT / "copy_binding_valid.jsonl"),
    "basic_arithmetic": (DATA_ROOT / "basic_arithmetic_train.jsonl", DATA_ROOT / "basic_arithmetic_valid.jsonl"),
    "epistemic_contrast": (DATA_ROOT / "epistemic_contrast_train.jsonl", DATA_ROOT / "epistemic_contrast_valid.jsonl"),
    "balanced_replay": (DATA_ROOT / "balanced_replay_train.jsonl", DATA_ROOT / "balanced_replay_valid.jsonl"),
}


def _row(user: str, assistant: str, family: str, skill: str, source: str = "generated") -> dict:
    return {
        "user": user.strip(),
        "assistant": assistant.strip(),
        "family": family,
        "skill": skill,
        "source": source,
    }


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _canonical_math(op: str, a: int, b: int) -> tuple[str, int, int]:
    if op in {"+", "*"}:
        a, b = sorted((a, b))
    return op, a, b


def _reserved_math(op: str, a: int, b: int) -> bool:
    target = _canonical_math(op, a, b)
    return any(_canonical_math(*x) == target for x in BENCHMARK_RESERVED_MATH)


def _reserved_false(op: str, a: int, b: int, wrong: int) -> bool:
    target = (*_canonical_math(op, a, b), wrong)
    return any((*_canonical_math(x[0], x[1], x[2]), x[3]) == target for x in BENCHMARK_RESERVED_FALSE_MATH)


def _dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        key = (normalize_surface(row["user"]), row["assistant"].casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out



def _filter_benchmark_surfaces(rows: list[dict]) -> list[dict]:
    held_out = benchmark_surface_prompts()
    return [row for row in rows if normalize_surface(row["user"]) not in held_out]


def _no_benchmark_surface(rows: list[dict]) -> list[str]:
    bench = benchmark_surface_prompts()
    return [r["user"] for r in rows if normalize_surface(r["user"]) in bench]


def _surface_family_overlap(train: list[dict], valid: list[dict]) -> set[str]:
    return {r["family"] for r in train} & {r["family"] for r in valid}


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _archive_previous_if_needed(target_version: str):
    if not MANIFEST_PATH.exists():
        return
    try:
        old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    old_target = str(old.get("target_version") or "unknown")
    if old_target == target_version:
        return
    files = [p for p in DATA_ROOT.glob("*.jsonl") if p.is_file()]
    if not files:
        return
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE_DIR / f"deliberate-v{old_target}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)
        z.write(MANIFEST_PATH, MANIFEST_PATH.name)
    for p in files:
        p.unlink(missing_ok=True)
    print(f"Archived previous deliberate corpus: {archive}")


def _concept_rows() -> tuple[list[dict], list[dict]]:
    concepts = {
        "archivo": [
            "Un archivo es una unidad donde la computadora guarda información o contenido.",
            "Un archivo guarda datos o contenido con un nombre dentro del almacenamiento.",
            "Pensalo como un contenedor de información guardada en la computadora.",
        ],
        "carpeta": [
            "Una carpeta sirve para organizar y contener archivos y otras carpetas.",
            "Una carpeta agrupa archivos para mantenerlos organizados en la computadora.",
            "Es un contenedor usado para ordenar archivos y subcarpetas.",
        ],
        "api": [
            "Una API es una interfaz que permite que programas o sistemas se comuniquen de forma definida.",
            "Una API define cómo una aplicación puede pedir datos o funciones a otro software.",
            "Es una forma acordada para que dos programas interactúen sin conocer todos sus detalles internos.",
        ],
        "parametro": [
            "Un parámetro de una red neuronal es un valor numérico ajustable que se aprende durante el entrenamiento.",
            "Es un número interno de la red que el entrenamiento modifica para mejorar sus predicciones.",
            "Los parámetros son valores aprendidos que influyen en cómo responde una red neuronal.",
        ],
        "token": [
            "Un token es una pieza de texto que el modelo procesa como una unidad.",
            "Un token puede ser una palabra, parte de una palabra o símbolo convertido en una unidad para el modelo.",
            "Es un fragmento de texto representado por un identificador que el modelo puede procesar.",
        ],
        "epoch": [
            "Una epoch es una pasada completa por el conjunto de datos de entrenamiento.",
            "Una epoch termina cuando el entrenamiento recorrió una vez todos los ejemplos del dataset.",
            "Es una vuelta completa sobre los datos usados para entrenar el modelo.",
        ],
        "dataset": [
            "Un dataset es un conjunto organizado de datos o ejemplos usados para entrenar, evaluar o analizar.",
            "Es una colección de ejemplos que se usa como datos de trabajo para entrenamiento o evaluación.",
            "Un dataset reúne datos de forma organizada para que puedan procesarse o aprenderse.",
        ],
        "ram": [
            "La RAM es memoria de trabajo temporal que guarda datos y programas mientras se están usando.",
            "La RAM mantiene temporalmente información que la computadora necesita rápido durante la ejecución.",
            "Es memoria rápida y temporal usada por los programas que están funcionando.",
        ],
        "cpu": [
            "La CPU es el procesador que ejecuta instrucciones y realiza cálculos para los programas.",
            "La CPU procesa instrucciones y coordina gran parte del trabajo general de la computadora.",
            "Es el procesador encargado de ejecutar operaciones e instrucciones de software.",
        ],
        "gpu": [
            "Una GPU es un procesador especializado en realizar muchas operaciones en paralelo, muy usado para gráficos y cómputo.",
            "La GPU puede ejecutar gran cantidad de cálculos paralelos y por eso también se usa para entrenar modelos.",
            "Es un procesador diseñado para trabajo paralelo, originalmente centrado en gráficos.",
        ],
        "proceso": [
            "Un proceso es una instancia de un programa que está ejecutándose.",
            "Cuando un programa se está ejecutando, el sistema operativo lo maneja como uno o más procesos.",
            "Es un programa en ejecución con sus propios recursos y estado.",
        ],
        "hilo": [
            "Un hilo es una secuencia de ejecución dentro de un proceso.",
            "Un proceso puede tener varios hilos que realizan distintas partes del trabajo.",
            "Es una unidad de ejecución que comparte recursos con otros hilos del mismo proceso.",
        ],
        "modelo": [
            "Un modelo de IA es una estructura matemática con parámetros que transforma entradas en salidas.",
            "Es un sistema de parámetros aprendidos que usa patrones del entrenamiento para producir resultados.",
            "Un modelo es la parte que procesa una entrada y genera una predicción o respuesta según lo aprendido.",
        ],
        "transformer": [
            "Un Transformer es una arquitectura de red neuronal que usa atención para relacionar partes de una secuencia.",
            "Es una arquitectura neuronal basada en mecanismos de atención para procesar secuencias como texto.",
            "Un Transformer permite que el modelo tenga en cuenta relaciones entre distintos tokens de una secuencia.",
        ],
    }

    train_templates = [
        ("def_train_01", "que es {c}"),
        ("def_train_02", "explicame {c} facil"),
        ("def_train_03", "{c} q significa"),
        ("def_train_04", "decime simple que seria {c}"),
        ("def_train_05", "no entiendo {c} explicamelo"),
        ("def_train_06", "para alguien nuevo q es {c}"),
        ("def_train_07", "en computacion que es {c}"),
        ("def_train_08", "me explicas {c} sin vueltas"),
        ("def_train_09", "{c} en una pc o ia que seria"),
        ("def_train_10", "quiero entender {c} en palabras faciles"),
    ]
    valid_templates = [
        ("def_valid_01", "si nunca programe como me explicarias {c}"),
        ("def_valid_02", "cuando alguien dice {c} a que se refiere"),
        ("def_valid_03", "haceme entender rapido el concepto {c}"),
        ("def_valid_04", "{c} vendria a ser que cosa"),
    ]

    train, valid = [], []
    for concept, answers in concepts.items():
        for fam, template in train_templates:
            for i, answer in enumerate(answers):
                train.append(_row(template.format(c=concept), answer, fam, f"definition:{concept}"))
                # casual surface variant stays in the same family
                casual = template.format(c=concept).replace("que ", "q ", 1).replace("explicame", "explicame")
                if casual != template.format(c=concept):
                    train.append(_row(casual, answers[(i + 1) % len(answers)], fam, f"definition:{concept}"))
        for fam, template in valid_templates:
            for i, answer in enumerate(answers[:2]):
                valid.append(_row(template.format(c=concept), answer, fam, f"definition:{concept}"))
    return train, valid


def _social_rows() -> tuple[list[dict], list[dict]]:
    specs = {
        "greeting": (
            ["hola", "buenas", "hey", "holaa", "buen dia", "que onda", "ey que tal", "buenass"],
            ["¡Hola! ¿En qué te puedo ayudar?", "Hola. ¿Qué necesitás?", "¡Buenas! Decime qué querés hacer."],
        ),
        "thanks": (
            ["gracias", "muchas gracias", "graciass", "me re ayudaste gracias", "joya gracias", "te agradezco"],
            ["De nada.", "¡De nada!", "Un placer."],
        ),
        "identity": (
            ["como te llamas", "cual es tu nombre", "quien sos", "vos quien sos", "decime tu nombre", "como es tu nombre"],
            ["Soy ButterflyAI.", "Me llamo ButterflyAI."],
        ),
        "state": (
            ["como estas", "todo bien", "como andas", "estas bien", "que tal estas", "todo tranqui por ahi"],
            ["Bien, gracias. Estoy lista para ayudarte.", "Todo bien. Estoy lista para ayudarte.", "Estoy funcionando bien y lista para ayudarte."],
        ),
        "clarify": (
            ["no entendi explicalo otra vez", "me perdi explicame de nuevo", "no se entendio nada", "decilo de otra forma"],
            ["Claro, te lo explico de otra forma.", "Sí, lo reformulo más simple.", "Te lo explico de nuevo paso a paso."],
        ),
        "goodbye": (
            ["chau", "nos vemos", "me fui", "hasta luego", "bueno chau", "nos vemoss"],
            ["¡Nos vemos!", "Hasta luego.", "¡Chau!"],
        ),
    }
    train, valid = [], []
    for skill, (prompts, answers) in specs.items():
        cut = max(2, len(prompts) - 2)
        for i, prompt in enumerate(prompts[:cut]):
            for j, answer in enumerate(answers):
                train.append(_row(prompt, answer, f"social_train_{skill}_{i % 3}", f"social:{skill}"))
        for i, prompt in enumerate(prompts[cut:]):
            for j, answer in enumerate(answers[:2]):
                valid.append(_row(prompt, answer, f"social_valid_{skill}_{i}", f"social:{skill}"))
    return train, valid


def build_robust_comprehension() -> tuple[list[dict], list[dict]]:
    a_train, a_valid = _concept_rows()
    s_train, s_valid = _social_rows()
    train = _dedupe(a_train + s_train)
    valid = _dedupe(a_valid + s_valid)
    return train, valid


def _make_copy_targets() -> tuple[list[str], list[str]]:
    rng = random.Random(SEED + 10)
    words = [
        "rojo", "verde", "amarillo", "violeta", "gris", "blanco", "negro", "lima", "coral", "ambar",
        "casa", "puerta", "mesa", "silla", "vaso", "libro", "lapiz", "papel", "bosque", "rio", "montana",
        "sol", "luna", "estrella", "viento", "lluvia", "fuego", "tierra", "piedra", "metal", "madera",
        "perro", "gato", "zorro", "tigre", "puma", "aguila", "halcon", "pez", "ballena", "delfin",
        "tren", "barco", "avion", "auto", "rueda", "motor", "puente", "camino", "plaza", "torre",
        "dato", "vector", "matriz", "modelo", "codigo", "pixel", "audio", "texto", "clave", "puerto",
    ]
    syllables = ["ba", "be", "bi", "bo", "ca", "ce", "ci", "da", "de", "di", "fa", "fi", "ga", "go", "la", "le", "li", "ma", "me", "mi", "na", "ne", "ni", "pa", "pe", "pi", "ra", "re", "ri", "sa", "se", "si", "ta", "te", "ti", "va", "ve", "vi", "zo"]
    generated = set()
    while len(generated) < 650:
        generated.add("".join(rng.choice(syllables) for _ in range(rng.choice([2, 3])))[:10])
    codes = set()
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    while len(codes) < 700:
        if rng.random() < 0.5:
            code = f"{rng.choice(alphabet)}{rng.randint(10,99)}{rng.choice(alphabet)}"
        else:
            code = f"{rng.choice(alphabet)}{rng.choice(alphabet)}{rng.randint(2,9)}"
        codes.add(code)
    values = list(dict.fromkeys(words + sorted(generated) + sorted(codes)))
    values = [x for x in values if x.casefold() not in BENCHMARK_RESERVED_EXACT_TARGETS]
    rng.shuffle(values)
    valid = values[:180]
    train = values[180:]
    assert not (set(x.casefold() for x in train) & set(x.casefold() for x in valid))
    return train, valid


def build_copy_binding() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED + 11)
    train_targets, valid_targets = _make_copy_targets()
    train_templates = [
        ("copy_train_01", "escribi solo {x}"),
        ("copy_train_02", "responde unicamente {x}"),
        ("copy_train_03", "pone {x} y nada mas"),
        ("copy_train_04", "tu salida debe ser {x}"),
        ("copy_train_05", "devolve exactamente {x}"),
        ("copy_train_06", "sin explicar contesta {x}"),
        ("copy_train_07", "quiero solo {x} sin texto extra"),
        ("copy_train_08", "respuesta: solamente {x}"),
        ("copy_train_09", "copia este valor como unica respuesta {x}"),
        ("copy_train_10", "no agregues nada escribi {x}"),
    ]
    valid_templates = [
        ("copy_valid_01", "contestame nada mas con {x}"),
        ("copy_valid_02", "la unica cosa que debes devolver es {x}"),
        ("copy_valid_03", "sin ningun agregado pone {x}"),
        ("copy_valid_04", "respuesta literal {x} solamente"),
        ("copy_valid_05", "deja como salida solo el valor {x}"),
    ]
    train, valid = [], []
    for target in train_targets:
        for fam, template in rng.sample(train_templates, 3):
            train.append(_row(template.format(x=target), target, fam, "binding"))
    for target in valid_targets:
        for fam, template in valid_templates:
            valid.append(_row(template.format(x=target), target, fam, "binding"))
    return _dedupe(train), _dedupe(valid)


def _math_result(op: str, a: int, b: int) -> int:
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    raise ValueError(op)


def _math_pairs() -> tuple[list[tuple[str,int,int]], list[tuple[str,int,int]]]:
    pairs = []
    for a in range(0, 21):
        for b in range(0, 21):
            pairs.append(("+", a, b))
    for a in range(0, 26):
        for b in range(0, a + 1):
            pairs.append(("-", a, b))
    for a in range(0, 13):
        for b in range(0, 13):
            pairs.append(("*", a, b))

    # Remove benchmark pairs including commutative mirrors.
    pairs = [p for p in pairs if not _reserved_math(*p)]
    train, valid = [], []
    seen_valid_canon = set()
    for p in pairs:
        canon = _canonical_math(*p)
        # Whole pair families are held out: for +/* both orientations go to the same side.
        is_valid = _stable_int(f"math:{canon}") % 11 == 0
        if is_valid:
            valid.append(p)
            seen_valid_canon.add(canon)
        else:
            train.append(p)
    # Defensive: no canonical train/valid overlap.
    assert not ({_canonical_math(*p) for p in train} & {_canonical_math(*p) for p in valid})
    return train, valid


def build_basic_arithmetic() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED + 12)
    train_pairs, valid_pairs = _math_pairs()
    train_templates = [
        ("math_train_01", "{a}{sym}{b}"),
        ("math_train_02", "cuanto es {a} {word} {b}"),
        ("math_train_03", "{a} {word} {b} solo resultado"),
        ("math_train_04", "resolve {a}{sym}{b} sin explicar"),
        ("math_train_05", "dame nada mas el numero de {a} {word} {b}"),
        ("math_train_06", "resultado de {a}{sym}{b}"),
        ("math_train_07", "calcula {a} {word} {b} y responde solo el numero"),
        ("math_train_08", "solo numero {a}{sym}{b}"),
    ]
    valid_templates = [
        ("math_valid_01", "decime el resultado exacto de {a} {word} {b}"),
        ("math_valid_02", "{a}{sym}{b} nada mas la respuesta"),
        ("math_valid_03", "sin texto extra cuanto da {a} {word} {b}"),
        ("math_valid_04", "respuesta numerica para {a}{sym}{b}"),
    ]
    words = {"+": "mas", "-": "menos", "*": "por"}
    train, valid = [], []
    for op, a, b in train_pairs:
        for fam, template in rng.sample(train_templates, 3):
            sym = "x" if op == "*" and _stable_int(f"train:{fam}:{a}:{b}") % 2 == 0 else op
            prompt = template.format(a=a, b=b, sym=sym, word=words[op])
            train.append(_row(prompt, str(_math_result(op, a, b)), fam, f"arithmetic:{op}"))
    for op, a, b in valid_pairs:
        for fam, template in valid_templates:
            sym = "x" if op == "*" and _stable_int(f"valid:{fam}:{a}:{b}") % 2 == 0 else op
            prompt = template.format(a=a, b=b, sym=sym, word=words[op])
            valid.append(_row(prompt, str(_math_result(op, a, b)), fam, f"arithmetic:{op}"))
    return _dedupe(train), _dedupe(valid)


def _fake_names(prefix: str, count: int, offset: int) -> list[str]:
    rng = random.Random(SEED + offset)
    starts = ["al", "bel", "cor", "dar", "el", "fal", "gor", "hel", "is", "jor", "kal", "lor", "mer", "nor", "or", "pel", "quil", "ras", "sel", "tor", "ur", "val", "wen", "yor", "zen"]
    ends = ["avia", "enia", "oria", "alia", "esia", "una", "ara", "eria", "onia", "iria", "ora", "istan", "ica", "uma"]
    out = set()
    while len(out) < count:
        out.add(prefix + rng.choice(starts) + rng.choice(ends))
    return sorted(x for x in out if x.casefold() not in BENCHMARK_RESERVED_FICTIONAL)


def build_epistemic_contrast() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED + 13)
    train, valid = [], []

    # Unknown fictional facts. Entity names are disjoint between train and validation.
    all_names = _fake_names("", 280, 31)
    train_names = all_names[:200]
    valid_names = all_names[200:260]
    train_templates = [
        ("unknown_train_01", "invente {x} pero no te dije su capital cual es"),
        ("unknown_train_02", "{x} es un pais inventado y nunca te conte su moneda cual usa"),
        ("unknown_train_03", "no te di ningun dato del lider de {x} quien es"),
        ("unknown_train_04", "si {x} me lo invente y no di ese dato podes saberlo"),
    ]
    valid_templates = [
        ("unknown_valid_01", "de {x} inventado no te conte ese dato que respondes"),
        ("unknown_valid_02", "sin informacion sobre {x} podes afirmar ese dato"),
    ]
    unknown_answers = [
        "No lo sé con los datos disponibles. Necesitaría que me des ese dato o poder verificarlo.",
        "No tengo información suficiente para saberlo; tendría que verificarlo o preguntarte.",
        "No puedo determinarlo sin datos o una fuente que lo confirme.",
    ]
    for name in train_names:
        for fam, template in rng.sample(train_templates, 2):
            train.append(_row(template.format(x=name), rng.choice(unknown_answers), fam, "epistemic:unknown"))
    for name in valid_names:
        for fam, template in valid_templates:
            valid.append(_row(template.format(x=name), rng.choice(unknown_answers), fam, "epistemic:unknown"))

    # False arithmetic claims: response must reject AND state the correct result.
    false_train_templates = [
        ("false_train_01", "me dicen que {a}{sym}{b}={wrong} esta bien"),
        ("false_train_02", "si alguien afirma {a} {word} {b} da {wrong} le crees"),
        ("false_train_03", "es verdad que {a}{sym}{b} es {wrong}"),
        ("false_train_04", "un amigo dice {a} {word} {b} = {wrong} correcto"),
    ]
    false_valid_templates = [
        ("false_valid_01", "che {a}{sym}{b} da {wrong} o esta mal"),
        ("false_valid_02", "me pasaron {a} {word} {b} como {wrong} verifica"),
    ]
    words = {"+": "mas", "-": "menos", "*": "por"}
    candidates = []
    for op, a, b in _math_pairs()[0][:]:
        correct = _math_result(op, a, b)
        wrong = correct + (1 if (_stable_int(str((op,a,b))) % 2 == 0) else -1)
        if wrong < 0 or _reserved_false(op, a, b, wrong):
            continue
        candidates.append((op, a, b, wrong, correct))
    rng.shuffle(candidates)
    for idx, (op, a, b, wrong, correct) in enumerate(candidates[:650]):
        fam, template = rng.choice(false_train_templates)
        prompt = template.format(a=a, b=b, sym=op, word=words[op], wrong=wrong)
        answer = f"No. {a} {op} {b} = {correct}."
        train.append(_row(prompt, answer, fam, "epistemic:false_math"))
    for idx, (op, a, b, wrong, correct) in enumerate(candidates[650:800]):
        fam, template = false_valid_templates[idx % len(false_valid_templates)]
        prompt = template.format(a=a, b=b, sym=op, word=words[op], wrong=wrong)
        answer = f"No. {a} {op} {b} = {correct}."
        valid.append(_row(prompt, answer, fam, "epistemic:false_math"))

    source_train = [
        ("source_train_01", "si aparece en internet ya es cierto", "No necesariamente. Hay que revisar la fuente, la evidencia y contrastar la información."),
        ("source_train_02", "una pagina lo dice entonces lo doy por verdadero", "No automáticamente. Conviene verificar la fuente y buscar evidencia independiente."),
        ("source_train_03", "dos fuentes confiables se contradicen que hago", "Compará las fuentes, fechas, contexto y evidencia antes de decidir cuál está mejor sustentada."),
        ("source_train_04", "si dos sitios serios dicen cosas distintas", "Hay que contrastar la evidencia, revisar el contexto y comprobar cuál afirmación está mejor respaldada."),
    ]
    source_valid = [
        ("source_valid_01", "algo viral en internet alcanza como prueba", "No. Una afirmación viral no reemplaza verificar la fuente y la evidencia."),
        ("source_valid_02", "hay evidencia seria que se contradice como seguis", "Revisaría y compararía las fuentes, el contexto, las fechas y la evidencia disponible."),
    ]
    for fam, prompt, answer in source_train:
        for suffix in ("", " decime corto", " sin inventar"):
            train.append(_row(prompt + suffix, answer, fam, "epistemic:sources"))
    for fam, prompt, answer in source_valid:
        valid.append(_row(prompt, answer, fam, "epistemic:sources"))

    return _dedupe(train), _dedupe(valid)


def _sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, n)


def _with_replay(base: list[dict], pools: list[tuple[list[dict], int]], seed: int) -> list[dict]:
    out = list(base)
    for i, (rows, n) in enumerate(pools):
        out.extend(_sample(rows, n, seed + i * 101))
    return _dedupe(out)


def build_all_rows() -> dict[str, tuple[list[dict], list[dict]]]:
    robust_raw = build_robust_comprehension()
    binding_raw = build_copy_binding()
    arithmetic_raw = build_basic_arithmetic()
    epistemic_raw = build_epistemic_contrast()
    robust = tuple(_filter_benchmark_surfaces(x) for x in robust_raw)
    binding = tuple(_filter_benchmark_surfaces(x) for x in binding_raw)
    arithmetic = tuple(_filter_benchmark_surfaces(x) for x in arithmetic_raw)
    epistemic = tuple(_filter_benchmark_surfaces(x) for x in epistemic_raw)

    # Rehearsal in later stages: new skill dominates, but old abilities never disappear
    # completely from the optimization stream.
    binding_train = _with_replay(binding[0], [(robust[0], 450)], SEED + 100)
    arithmetic_train = _with_replay(arithmetic[0], [(robust[0], 350), (binding[0], 500)], SEED + 200)
    epistemic_train = _with_replay(epistemic[0], [(robust[0], 300), (binding[0], 350), (arithmetic[0], 350)], SEED + 300)

    mixed_train = _dedupe(
        _sample(robust[0], 1800, SEED + 401)
        + _sample(binding[0], 2200, SEED + 402)
        + _sample(arithmetic[0], 2200, SEED + 403)
        + _sample(epistemic[0], 1800, SEED + 404)
    )
    mixed_valid = _dedupe(
        _sample(robust[1], 350, SEED + 411)
        + _sample(binding[1], 350, SEED + 412)
        + _sample(arithmetic[1], 350, SEED + 413)
        + _sample(epistemic[1], 300, SEED + 414)
    )

    return {
        "robust_comprehension": robust,
        "copy_binding": (binding_train, binding[1]),
        "basic_arithmetic": (arithmetic_train, arithmetic[1]),
        "epistemic_contrast": (epistemic_train, epistemic[1]),
        "balanced_replay": (mixed_train, mixed_valid),
    }


def _validate_all(rows_by_stage: dict[str, tuple[list[dict], list[dict]]]):
    leaks = []
    overlaps = {}
    benchmark_values = {x.casefold() for x in BENCHMARK_RESERVED_EXACT_TARGETS}
    for stage, (train, valid) in rows_by_stage.items():
        leaks.extend((stage, x) for x in _no_benchmark_surface(train + valid))
        overlap = _surface_family_overlap(train, valid)
        if overlap:
            overlaps[stage] = sorted(overlap)
        for row in train + valid:
            if row["skill"] == "binding" and row["assistant"].casefold() in benchmark_values:
                raise RuntimeError(f"Reserved exact target leaked into {stage}: {row['assistant']}")
            low = row["user"].casefold()
            for fictional in BENCHMARK_RESERVED_FICTIONAL:
                if re.search(rf"\b{re.escape(fictional)}\b", low):
                    raise RuntimeError(f"Reserved fictional entity leaked into {stage}: {fictional}")
    if leaks:
        preview = "\n".join(f"{s}: {p}" for s, p in leaks[:10])
        raise RuntimeError("Benchmark surface leak detected:\n" + preview)
    if overlaps:
        raise RuntimeError(f"Train/valid family overlap: {overlaps}")


def build_corpus(target_version: str) -> dict:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _archive_previous_if_needed(target_version)
    rows_by_stage = build_all_rows()
    _validate_all(rows_by_stage)

    manifest = {
        "format": 1,
        "target_version": target_version,
        "benchmark_suite": BENCHMARK_SUITE_VERSION,
        "seed": SEED,
        "benchmark_surface_leaks": 0,
        "train_valid_surface_family_overlap": 0,
        "reserved_exact_targets": sorted(BENCHMARK_RESERVED_EXACT_TARGETS),
        "reserved_math_pairs": sorted([list(x) for x in BENCHMARK_RESERVED_MATH]),
        "reserved_false_math": sorted([list(x) for x in BENCHMARK_RESERVED_FALSE_MATH]),
        "reserved_fictional": sorted(BENCHMARK_RESERVED_FICTIONAL),
        "stages": {},
    }

    print(f"ButterflyAI v{target_version} deliberate corpus built.")
    print(f"Benchmark suite held out       : v{BENCHMARK_SUITE_VERSION}")
    print("Benchmark surface leaks        : 0")
    print("Train/valid surface-family overlap: 0")
    print("Exact-copy benchmark targets   : RESERVED, never trained/validated")
    print("Arithmetic benchmark pairs     : RESERVED, never trained/validated")

    for stage, (train, valid) in rows_by_stage.items():
        train_path, valid_path = STAGE_FILES[stage]
        _write_jsonl(train_path, train)
        _write_jsonl(valid_path, valid)
        train_fams = len({r["family"] for r in train})
        valid_fams = len({r["family"] for r in valid})
        train_skills = Counter(r["skill"] for r in train)
        manifest["stages"][stage] = {
            "train_rows": len(train),
            "valid_rows": len(valid),
            "train_families": train_fams,
            "valid_families": valid_fams,
            "train_skills": dict(train_skills),
            "train_file": str(train_path.relative_to(ROOT)),
            "valid_file": str(valid_path.relative_to(ROOT)),
            "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "valid_sha256": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
        }
        mb = train_path.stat().st_size / (1024 * 1024)
        print(f"{stage:22s} train {len(train):,} rows | valid {len(valid):,} | families {train_fams}/{valid_fams} | {mb:.2f} MB train")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    history = {"format": 1, "builds": []}
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    history["builds"] = [x for x in history.get("builds", []) if x.get("target_version") != target_version]
    history["builds"].append({
        "target_version": target_version,
        "benchmark_suite": BENCHMARK_SUITE_VERSION,
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "stage_rows": {k: {"train": v["train_rows"], "valid": v["valid_rows"]} for k, v in manifest["stages"].items()},
    })
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {MANIFEST_PATH}")
    return manifest


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing deliberate corpus manifest: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
