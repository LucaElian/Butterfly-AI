from __future__ import annotations

import random
from .common import row


def _make_code(rng: random.Random, shape: str) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    digits = "23456789"
    out = []
    for token in shape:
        if token == "L":
            out.append(rng.choice(letters))
        elif token == "D":
            out.append(rng.choice(digits))
        else:
            raise ValueError(shape)
    return "".join(out)


def build(seed: int) -> tuple[list[dict], list[dict]]:
    from ...learning.evaluator import BENCHMARK_RESERVED_EXACT_TARGETS
    from ...learning.study_exam import STUDY_RESERVED_EXACT_TARGETS

    rng = random.Random(seed)
    words = {
        "rojo", "verde", "amarillo", "violeta", "gris", "blanco", "negro",
        "casa", "puerta", "mesa", "silla", "vaso", "libro", "bosque", "rio",
        "sol", "luna", "estrella", "viento", "lluvia", "fuego", "tierra",
        "dato", "vector", "matriz", "modelo", "codigo", "pixel", "audio", "texto",
        "bronce", "plata", "arena", "piedra", "rama", "hoja", "motor", "rueda",
        "ventana", "tecla", "cable", "nodo", "ruta", "puerto", "cache", "registro",
        "delta", "sigma", "omega", "nexo", "lumen", "orbita", "trama", "pulso",
    }
    shapes = (
        "LDDL", "LDDDL", "LLD", "LLDD", "LDL", "LDLD",
        "LLDL", "LDDD", "DDLL", "DLLD", "LLLDD", "LDDLL",
    )
    generated = set(words)
    while len(generated) < 2200:
        generated.add(_make_code(rng, rng.choice(shapes)))

    reserved = {
        str(x).casefold()
        for x in (set(BENCHMARK_RESERVED_EXACT_TARGETS) | set(STUDY_RESERVED_EXACT_TARGETS))
    }
    values = [x for x in generated if x.casefold() not in reserved]
    rng.shuffle(values)

    valid_values = values[:360]
    train_values = values[360:]

    train_templates = [
        "escribi solo {x}", "responde unicamente {x}", "pone {x} y nada mas",
        "tu salida debe ser {x}", "devolve exactamente {x}", "sin explicar contesta {x}",
        "solo quiero {x}", "responde con {x} sin agregar texto",
        "escribi exactamente {x} y termina", "contesta nada mas {x}",
    ]
    valid_templates = [
        "la unica respuesta tiene que ser {x}", "sin ningun agregado devolve {x}",
        "quiero exactamente {x} como salida", "pone solamente {x} y nada mas",
    ]

    train, valid = [], []
    for target_index, target in enumerate(train_values):
        for template_index, template in enumerate(rng.sample(train_templates, 4)):
            train.append(
                row(
                    template.format(x=target),
                    target,
                    f"binding:train:{target_index}:{template_index}",
                    "binding",
                )
            )
    for target_index, target in enumerate(valid_values):
        for template_index, template in enumerate(valid_templates):
            valid.append(
                row(
                    template.format(x=target),
                    target,
                    f"binding:valid:{target_index}:{template_index}",
                    "binding",
                )
            )
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid
