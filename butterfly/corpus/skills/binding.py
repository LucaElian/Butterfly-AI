from __future__ import annotations

import random
from .common import row


def build(seed: int) -> tuple[list[dict], list[dict]]:
    from ...learning.evaluator import BENCHMARK_RESERVED_EXACT_TARGETS

    rng = random.Random(seed)
    words = [
        "rojo", "verde", "amarillo", "violeta", "gris", "blanco", "negro",
        "casa", "puerta", "mesa", "silla", "vaso", "libro", "bosque", "rio",
        "sol", "luna", "estrella", "viento", "lluvia", "fuego", "tierra",
        "dato", "vector", "matriz", "modelo", "codigo", "pixel", "audio", "texto",
    ]
    generated = set(words)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    while len(generated) < 900:
        generated.add(f"{rng.choice(alphabet)}{rng.randint(10, 999)}{rng.choice(alphabet)}")
    values = [x for x in generated if x.casefold() not in BENCHMARK_RESERVED_EXACT_TARGETS]
    rng.shuffle(values)
    valid_values = values[:160]
    train_values = values[160:]

    train_templates = [
        "escribi solo {x}", "responde unicamente {x}", "pone {x} y nada mas",
        "tu salida debe ser {x}", "devolve exactamente {x}", "sin explicar contesta {x}",
    ]
    valid_templates = [
        "contestame nada mas con {x}", "la unica cosa que debes devolver es {x}",
        "sin ningun agregado pone {x}",
    ]
    train, valid = [], []
    for target in train_values:
        for i, template in enumerate(rng.sample(train_templates, 3)):
            train.append(row(template.format(x=target), target, f"binding:train:{i}", "binding"))
    for target in valid_values:
        for i, template in enumerate(valid_templates):
            valid.append(row(template.format(x=target), target, f"binding:valid:{i}", "binding"))
    return train, valid
