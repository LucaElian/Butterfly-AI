from __future__ import annotations

import random
from .common import row


def build(seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    concepts = {
        "archivo": [
            "Un archivo guarda información o contenido en la computadora.",
            "Un archivo es una unidad de información guardada con un nombre.",
        ],
        "carpeta": [
            "Una carpeta organiza y contiene archivos y otras carpetas.",
            "Una carpeta agrupa archivos para mantenerlos organizados.",
        ],
        "api": [
            "Una API permite que programas o sistemas se comuniquen de forma definida.",
            "Una API define cómo un programa puede pedir datos o funciones a otro software.",
        ],
        "parametro": [
            "Un parámetro es un valor numérico interno que la red ajusta durante el entrenamiento.",
            "Es un número aprendido que influye en cómo responde una red neuronal.",
        ],
        "token": [
            "Un token es una pieza de texto que el modelo procesa como una unidad.",
            "Puede ser una palabra, parte de una palabra o símbolo tratado como una unidad.",
        ],
        "dataset": [
            "Un dataset es un conjunto organizado de datos o ejemplos usados para entrenar o evaluar.",
            "Es una colección de ejemplos que se usa como datos de trabajo.",
        ],
        "epoch": [
            "Una epoch es una pasada completa por los datos de entrenamiento.",
            "Es una vuelta completa sobre el dataset durante el entrenamiento.",
        ],
        "ram": [
            "La RAM es memoria temporal de trabajo usada por los programas mientras funcionan.",
            "La RAM mantiene datos temporales que la computadora necesita acceder rápido.",
        ],
        "cpu": [
            "La CPU es el procesador que ejecuta instrucciones y realiza cálculos.",
            "La CPU procesa instrucciones y coordina gran parte del trabajo general de la computadora.",
        ],
    }
    train_templates = [
        "que es {c}", "explicame {c} facil", "{c} q significa",
        "decime simple que seria {c}", "para alguien nuevo q es {c}",
        "me explicas {c} sin vueltas",
    ]
    valid_templates = [
        "si nunca programe como me explicarias {c}",
        "cuando alguien dice {c} a que se refiere",
        "{c} vendria a ser que cosa",
    ]
    train, valid = [], []
    for concept, answers in concepts.items():
        for i, template in enumerate(train_templates):
            for answer in answers:
                train.append(row(
                    template.format(c=concept), answer,
                    f"concept:{concept}:train:{i}", f"definition:{concept}"
                ))
        for i, template in enumerate(valid_templates):
            valid.append(row(
                template.format(c=concept), rng.choice(answers),
                f"concept:{concept}:valid:{i}", f"definition:{concept}"
            ))
    return train, valid
