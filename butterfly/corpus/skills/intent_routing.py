from __future__ import annotations

import random
from .common import row


def build(seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    specs = {
        "greeting": {
            "train": ["hola", "buenas", "hey", "holaa", "buen dia", "que onda", "ey que tal", "buenass", "holis"],
            "valid": ["buenass che", "eyy hola", "que onda todo bien"],
            "answers": ["¡Hola! ¿En qué te puedo ayudar?", "Hola. ¿Qué necesitás?", "¡Buenas! Decime qué querés hacer."],
        },
        "thanks": {
            "train": ["gracias", "muchas gracias", "graciass", "me re ayudaste gracias", "joya gracias", "te agradezco", "mil gracias"],
            "valid": ["gracias che me sirvio", "posta gracias", "te agradezco una banda"],
            "answers": ["De nada.", "¡De nada!", "Un placer."],
        },
        "identity": {
            "train": ["como te llamas", "cual es tu nombre", "quien sos", "vos quien sos", "decime tu nombre", "como es tu nombre"],
            "valid": ["quien vendrias a ser vos", "tu nombre cual es", "como te dicen"],
            "answers": ["Soy ButterflyAI.", "Me llamo ButterflyAI."],
        },
        "state": {
            "train": ["como estas", "todo bien", "como andas", "estas bien", "que tal estas", "todo tranqui por ahi", "como venis"],
            "valid": ["y vos como andas", "todo bien de tu lado", "como venis hoy"],
            "answers": ["Bien, gracias. Estoy lista para ayudarte.", "Todo bien. Estoy lista para ayudarte.", "Estoy funcionando bien y lista para ayudarte."],
        },
        "file": {
            "train": ["que es un archivo", "archivo q es", "para q sirve un archivo", "explicame archivo", "archivo en pc q seria"],
            "valid": ["que vendria a ser un archivo", "un archivo en la compu q es", "archivo explicamelo facil"],
            "answers": [
                "Un archivo guarda información o contenido en la computadora.",
                "Un archivo es una unidad donde la computadora guarda datos o contenido.",
            ],
        },
        "folder": {
            "train": ["que es una carpeta", "carpeta q es", "para q sirve una carpeta", "explicame carpeta", "carpeta de pc q seria"],
            "valid": ["una carpeta para q sirve", "que vendria a ser una carpeta", "carpeta explicamela facil"],
            "answers": [
                "Una carpeta organiza y contiene archivos y otras carpetas.",
                "Una carpeta sirve para agrupar archivos y mantenerlos organizados.",
            ],
        },
        "api": {
            "train": ["que es una api", "api q es", "para q sirve una api", "explicame api", "api que hace"],
            "valid": ["una api para que sirve", "api vendria a ser que", "explicame que hace una api"],
            "answers": [
                "Una API permite que programas o sistemas se comuniquen de forma definida.",
                "Una API define cómo un programa puede pedir datos o funciones a otro software.",
            ],
        },
        "parameter": {
            "train": ["que es un parametro en una red neuronal", "parametro de red neuronal q es", "explicame parametro en ia", "que significa parametro en una red"],
            "valid": ["en una red neuronal q seria un parametro", "parametro de ia explicamelo", "que hace un parametro de una red"],
            "answers": [
                "Es un número interno que la red ajusta durante el entrenamiento.",
                "Un parámetro es un valor numérico que la red aprende y modifica al entrenarse.",
            ],
        },
        "token": {
            "train": ["que es un token en ia", "token q es", "explicame token", "token en modelo q significa"],
            "valid": ["token q seria en un modelo", "cuando dicen token en ia que es", "token explicamelo facil"],
            "answers": [
                "Un token es una pieza de texto que el modelo procesa como una unidad.",
                "Un token es una unidad de texto, como una palabra, parte de palabra o símbolo.",
            ],
        },
        "clarify": {
            "train": ["no entendi explicalo otra vez", "me perdi explicame de nuevo", "no se entendio", "decilo de otra forma"],
            "valid": ["explicalo distinto que no entendi", "me lo repetis mas simple", "no lo capte reformulalo"],
            "answers": ["Claro, te lo explico de otra forma.", "Sí, lo reformulo más simple.", "Te lo explico de nuevo paso a paso."],
        },
        "goodbye": {
            "train": ["chau", "nos vemos", "me fui", "hasta luego", "bueno chau"],
            "valid": ["nos vemoss", "listo chau che", "hasta la proxima"],
            "answers": ["¡Nos vemos!", "Hasta luego.", "¡Chau!"],
        },
    }

    train, valid = [], []
    train_prefixes = ["", "che ", "ey ", "a ver ", "decime "]
    train_suffixes = ["", " porfa", " sin vueltas", " explicame simple"]
    valid_prefixes = ["", "che ", "posta "]
    valid_suffixes = ["", " rapido", " facil"]

    for intent, spec in specs.items():
        for i, prompt in enumerate(spec["train"]):
            for prefix in train_prefixes:
                for suffix in train_suffixes:
                    train.append(row(
                        (prefix + prompt + suffix).strip(),
                        rng.choice(spec["answers"]),
                        f"route:{intent}:train:{i}",
                        f"intent:{intent}",
                    ))
        for i, prompt in enumerate(spec["valid"]):
            for prefix in valid_prefixes:
                for suffix in valid_suffixes:
                    valid.append(row(
                        (prefix + prompt + suffix).strip(),
                        rng.choice(spec["answers"]),
                        f"route:{intent}:valid:{i}",
                        f"intent:{intent}",
                    ))
    return train, valid
