from __future__ import annotations

import random
from .common import row


def _names(rng, count):
    starts = [
        "al", "bel", "cor", "dar", "el", "fal", "gor", "hel", "is", "jor", "kal",
        "lor", "mer", "nor", "pel", "quil", "ras", "sel", "tor", "val", "yor", "zen",
    ]
    ends = ["avia", "enia", "oria", "alia", "esia", "una", "ara", "eria", "onia", "iria", "ora", "ica", "uma"]
    out = set()
    while len(out) < count:
        out.add(rng.choice(starts) + rng.choice(ends))
    return sorted(out)


def build(seed: int) -> tuple[list[dict], list[dict]]:
    from ...learning.evaluator import BENCHMARK_RESERVED_FICTIONAL

    rng = random.Random(seed)
    names = [x for x in _names(rng, 280) if x.casefold() not in BENCHMARK_RESERVED_FICTIONAL]
    train_names = names[:190]
    valid_names = names[190:250]
    answers = [
        "No lo sé con los datos disponibles. Necesitaría que me des ese dato o poder verificarlo.",
        "No tengo información suficiente para saberlo; tendría que verificarlo o preguntarte.",
        "No puedo determinarlo sin datos o una fuente que lo confirme.",
    ]
    train_templates = [
        "invente {x} pero no te dije su capital cual es",
        "{x} es inventado y nunca te conte su moneda cual usa",
        "no te di ningun dato del lider de {x} quien es",
    ]
    valid_templates = [
        "de {x} inventado no te conte ese dato que respondes",
        "sin informacion sobre {x} podes afirmar ese dato",
    ]
    train, valid = [], []
    for name in train_names:
        for i, template in enumerate(train_templates):
            train.append(row(
                template.format(x=name), rng.choice(answers),
                f"unknown:train:{i}", "epistemic:unknown"
            ))
    for name in valid_names:
        for i, template in enumerate(valid_templates):
            valid.append(row(
                template.format(x=name), rng.choice(answers),
                f"unknown:valid:{i}", "epistemic:unknown"
            ))

    source_train = [
        ("si aparece en internet ya es cierto", "No necesariamente. Hay que revisar la fuente, la evidencia y contrastar la información."),
        ("una pagina lo dice entonces lo doy por verdadero", "No automáticamente. Conviene verificar la fuente y buscar evidencia independiente."),
        ("dos fuentes confiables se contradicen que hago", "Compará las fuentes, fechas, contexto y evidencia antes de decidir cuál está mejor sustentada."),
    ]
    source_valid = [
        ("algo viral en internet alcanza como prueba", "No. Una afirmación viral no reemplaza verificar la fuente y la evidencia."),
        ("hay evidencia seria que se contradice como seguis", "Revisaría y compararía las fuentes, el contexto, las fechas y la evidencia disponible."),
    ]
    for i, (prompt, answer) in enumerate(source_train):
        for suffix in ("", " decime corto", " sin inventar"):
            train.append(row(prompt + suffix, answer, f"sources:train:{i}", "epistemic:sources"))
    for i, (prompt, answer) in enumerate(source_valid):
        valid.append(row(prompt, answer, f"sources:valid:{i}", "epistemic:sources"))
    return train, valid
