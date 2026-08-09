from __future__ import annotations
import random


def build_curriculum(limit: int = 600, seed: int = 23):
    prompts: list[str] = []

    # ~40% normal language/conversation. Butterfly must learn to talk before it learns to work.
    casual = [
        "Hola", "Buenas", "Hola Butterfly", "Como estas?", "Que tal?", "Buen dia", "Buenas noches",
        "Gracias", "De nada", "Que haces?", "Que estas diciendo?", "No entendi, explicalo de otra forma",
        "Jajaja", "Estoy cansado", "Estoy aburrido", "Tengo una duda", "Me ayudas con algo?", "Dale",
        "Si", "No", "Puede ser", "Contame algo corto", "Presentate", "Quien sos?", "Como te llamas?",
        "Que significa aprender?", "Explicame algo como si fuera principiante", "Podes responder mas corto?",
        "Podes darme un ejemplo?", "No era eso lo que queria", "Corregi tu respuesta anterior",
    ]
    prompts.extend(casual * 3)

    everyday_topics = [
        "una casa", "una escuela", "un trabajo", "una computadora", "un telefono", "un libro", "una pelicula",
        "un juego", "una receta", "una ciudad", "un viaje", "el clima", "una amistad", "una mascota", "un auto",
        "un tren", "una bicicleta", "un examen", "una compra", "una planta", "una fotografia", "una cancion",
    ]
    for topic in everyday_topics:
        prompts += [
            f"Escribi dos oraciones naturales sobre {topic}.",
            f"Formula una pregunta cotidiana relacionada con {topic} y respondela brevemente.",
            f"Explica {topic} con palabras simples.",
        ]

    concepts = [
        "archivo", "carpeta", "programa", "memoria RAM", "CPU", "GPU", "base de datos", "algoritmo", "variable",
        "funcion", "clase", "API", "Excel", "PDF", "navegador", "sistema operativo", "red neuronal",
        "modelo de lenguaje", "token", "entrenamiento", "hipotesis", "evidencia", "hecho", "inferencia",
        "fuente confiable", "copia de seguridad", "compilador", "proceso", "ventana", "pestana", "JSON", "SQLite",
    ]
    for c in concepts:
        prompts += [
            f"Explica que es {c} en pocas palabras y da un ejemplo sencillo.",
            f"Da una definicion clara de {c} sin usar tecnicismos innecesarios.",
        ]

    for a in range(2, 20, 3):
        for b in (2, 5, 7):
            prompts += [
                f"Cuanto es {a} + {b}? Responde y explica brevemente.",
                f"Cuanto es {a} * {b}? Responde y verifica el calculo.",
            ]

    reasoning = [
        "Si Ana tiene tres manzanas y compra dos mas, cuantas tiene?",
        "Todos los perros son mamiferos. Firulais es un perro. Que podes concluir?",
        "Tengo que elegir entre un metodo rapido pero inseguro y uno un poco mas lento con verificacion. Cual conviene para datos importantes?",
        "Si una instruccion es ambigua, que deberias hacer antes de modificar archivos?",
        "Si una prueba falla despues de cambiar codigo, que indica eso?",
        "Si dos metodos producen resultados distintos, como investigarias cual es correcto?",
    ]
    prompts.extend(reasoning * 3)

    epistemic = [
        "Si te digo que 2 + 2 = 5, que deberias hacer?",
        "Que haces si un dato necesario no aparece en los archivos disponibles?",
        "Como diferencias un hecho de una suposicion?",
        "Que deberias hacer si dos fuentes confiables se contradicen?",
        "Por que no conviene creer automaticamente todo lo que aparece en Internet?",
        "Si no estas segura de una respuesta factual, como deberias responder?",
        "Explica por que una experiencia fallida puede servir para aprender.",
        "Antes de modificar cientos de archivos, que deberias comprobar?",
        "Despues de terminar una tarea, que cosas deberias verificar?",
        "Cuando conviene pedirle una aclaracion al usuario?",
        "Como elegis entre verificar con Internet, con una calculadora, con un compilador o con los propios archivos?",
    ]
    prompts.extend(epistemic * 3)

    pc_tasks = [
        "Abrir Bloc de notas, escribir un texto, guardarlo en Descargas y volver a abrirlo.",
        "Revisar veinte archivos Excel y detectar filas con datos faltantes.",
        "Buscar un archivo por nombre sin recorrer visualmente cada carpeta.",
        "Cambiar entre dos pestanas de un navegador de forma eficiente.",
        "Leer miles de celdas de Excel de la forma mas eficiente posible.",
        "Compilar un proyecto C# y revisar sus errores.",
        "Ordenar una carpeta grande de documentos sin perder archivos.",
        "Comparar dos versiones de un documento y resumir las diferencias.",
        "Crear una copia de seguridad antes de una modificacion masiva.",
        "Encontrar una estrategia mas rapida para una tarea repetitiva.",
    ]
    for task in pc_tasks:
        prompts += [f"Propone un plan breve y seguro para esta tarea: {task}", f"Que verificarias antes y despues de hacer esto: {task}"]

    coding = [
        "Que hace un bucle for? Da un ejemplo corto en Python.", "Explica la diferencia entre una lista y un diccionario.",
        "Que significa que un programa compile?", "Que es una excepcion y por que conviene manejarla?",
        "Por que deberia ejecutar tests despues de modificar codigo?", "Como investigarias un error sin inventar la causa?",
        "Que ventaja tiene automatizar una tarea que se repite mucho?", "Que es JSON y para que sirve?",
        "Que es SQLite?", "Explica que es una API con un ejemplo cotidiano.",
    ]
    prompts.extend(coding * 2)

    transformations = [
        ("Resume", "Un buen asistente debe comprobar los datos importantes antes de afirmar que una tarea esta terminada."),
        ("Reescribe de forma mas clara", "si falta info no inventar cosas y preguntar o buscar"),
        ("Convierte en una regla corta", "Antes de borrar muchos archivos hay que tener una forma de recuperarlos."),
        ("Explica como a un principiante", "Un modelo de lenguaje predice tokens basandose en contexto."),
        ("Da un ejemplo", "Una inferencia no es lo mismo que un hecho comprobado."),
    ]
    for instruction, text in transformations:
        prompts.extend([f"{instruction}: {text}"] * 3)

    objects = ["un archivo", "una planilla", "un proyecto", "un informe", "una carpeta"]
    actions = ["revisar", "organizar", "comparar", "validar", "resumir"]
    for action in actions:
        for obj in objects:
            prompts.append(f"Si te pido {action} {obj}, que preguntas o comprobaciones harias antes de empezar?")

    rnd = random.Random(seed)
    rnd.shuffle(prompts)
    if len(prompts) < limit:
        # Generate harmless paraphrase requests to reach the requested count.
        base = list(prompts)
        while len(prompts) < limit:
            p = rnd.choice(base)
            prompts.append(f"Responde de una manera natural y diferente a esta consigna: {p}")
    return prompts[:limit]
