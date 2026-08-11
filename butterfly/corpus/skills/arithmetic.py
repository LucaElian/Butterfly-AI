from __future__ import annotations

import random
from .common import row, stable_int


def _canonical(op: str, a: int, b: int):
    if op in {"+", "*"}:
        a, b = sorted((a, b))
    return op, a, b


def _result(op: str, a: int, b: int):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(op)


def build(seed: int) -> tuple[list[dict], list[dict]]:
    from ...learning.evaluator import BENCHMARK_RESERVED_MATH

    reserved = {_canonical(*x) for x in BENCHMARK_RESERVED_MATH}
    pairs = []
    for a in range(0, 31):
        for b in range(0, 31):
            pairs.append(("+", a, b))
    for a in range(0, 31):
        for b in range(0, a + 1):
            pairs.append(("-", a, b))
    for a in range(0, 13):
        for b in range(0, 13):
            pairs.append(("*", a, b))
    pairs = [p for p in pairs if _canonical(*p) not in reserved]

    train_pairs, valid_pairs = [], []
    for pair in pairs:
        if stable_int(f"math:{_canonical(*pair)}:{seed}") % 11 == 0:
            valid_pairs.append(pair)
        else:
            train_pairs.append(pair)

    rng = random.Random(seed)
    words = {"+": "mas", "-": "menos", "*": "por"}
    train_templates = [
        "{a}{sym}{b}", "cuanto es {a} {word} {b}", "{a} {word} {b} solo resultado",
        "resolve {a}{sym}{b} sin explicar", "solo numero {a}{sym}{b}",
    ]
    valid_templates = [
        "decime el resultado exacto de {a} {word} {b}",
        "{a}{sym}{b} nada mas la respuesta",
        "sin texto extra cuanto da {a} {word} {b}",
    ]
    train, valid = [], []
    for op, a, b in train_pairs:
        for i, template in enumerate(rng.sample(train_templates, 3)):
            sym = "x" if op == "*" and (a + b + i) % 2 == 0 else op
            train.append(row(
                template.format(a=a, b=b, sym=sym, word=words[op]),
                str(_result(op, a, b)),
                f"math:train:{i}",
                f"arithmetic:{op}",
            ))
    for op, a, b in valid_pairs:
        for i, template in enumerate(valid_templates):
            sym = "x" if op == "*" and (a + b + i) % 2 == 0 else op
            valid.append(row(
                template.format(a=a, b=b, sym=sym, word=words[op]),
                str(_result(op, a, b)),
                f"math:valid:{i}",
                f"arithmetic:{op}",
            ))
    return train, valid
