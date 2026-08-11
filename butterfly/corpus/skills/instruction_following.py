from __future__ import annotations

import random

from .common import row


TRAIN_FACTS = [
    ("mouse", "Un mouse permite mover el puntero y seleccionar elementos en la computadora."),
    ("teclado", "Un teclado permite escribir texto y ejecutar comandos mediante sus teclas."),
    ("monitor", "Un monitor muestra visualmente la información que genera la computadora."),
    ("backup", "Un backup es una copia de datos guardada para poder recuperarlos si se pierden."),
    ("navegador", "Un navegador es un programa usado para abrir y recorrer sitios web."),
    ("variable", "Una variable guarda un valor que un programa puede consultar o modificar."),
    ("base de datos", "Una base de datos organiza información para poder guardarla y consultarla."),
    ("contraseña", "Una contraseña es una credencial secreta usada para proteger un acceso."),
    ("actualizacion", "Una actualización instala cambios que pueden corregir errores o agregar mejoras."),
    ("proceso", "Un proceso es un programa o tarea que se está ejecutando en el sistema."),
    ("registro", "Un registro guarda eventos o datos para poder consultarlos después."),
    ("puerto", "Un puerto identifica un punto de comunicación usado por un servicio o programa."),
    ("cache", "Una caché guarda temporalmente datos para reutilizarlos con mayor rapidez."),
    ("ruta", "Una ruta indica la ubicación de un elemento dentro del sistema."),
    ("servidor", "Un servidor ofrece datos o servicios a otros equipos o programas que los solicitan."),
    ("cliente", "Un cliente solicita datos o funciones a otro sistema o servicio."),
    ("dependencia", "Una dependencia es un componente que otro programa necesita para funcionar."),
    ("configuracion", "Una configuración reúne valores que controlan cómo funciona un programa."),
    ("extension", "Una extensión ayuda a identificar el tipo o formato de un elemento."),
    ("directorio", "Un directorio organiza elementos dentro del sistema."),
    ("cola", "Una cola organiza elementos para atenderlos normalmente en el orden en que llegan."),
    ("indice", "Un índice organiza referencias para facilitar una búsqueda."),
    ("sesion", "Una sesión representa un período de interacción asociado a un usuario o proceso."),
    ("variable de entorno", "Una variable de entorno guarda un valor disponible para procesos del sistema."),
    ("paquete", "Un paquete agrupa componentes que se distribuyen juntos."),
    ("servicio", "Un servicio puede ejecutarse en segundo plano para ofrecer una función."),
    ("acceso directo", "Un acceso directo apunta a otro elemento para poder abrirlo rápidamente."),
    ("log", "Un log registra eventos para poder revisar lo que ocurrió."),
    ("checksum", "Un checksum es un valor usado para detectar cambios en datos."),
    ("branch", "Una branch permite trabajar en una línea separada de cambios."),
    ("commit", "Un commit registra un conjunto concreto de cambios en un repositorio."),
    ("socket", "Un socket representa un punto de comunicación entre procesos o sistemas."),
    ("request", "Una request es una solicitud enviada a otro componente o servicio."),
    ("response", "Una response es la respuesta devuelta después de una solicitud."),
    ("cache local", "Una caché local conserva temporalmente resultados para reutilizarlos."),
    ("permiso", "Un permiso define qué acciones están autorizadas sobre un recurso."),
]

VALID_FACTS = [
    ("daemon", "Un daemon es un proceso que puede ejecutarse en segundo plano."),
    ("proxy", "Un proxy actúa como intermediario entre una solicitud y su destino."),
    ("endpoint", "Un endpoint identifica un punto concreto donde se ofrece una función."),
    ("repositorio", "Un repositorio almacena y organiza versiones de un proyecto."),
    ("parche", "Un parche reúne cambios destinados a corregir o modificar software."),
    ("release", "Una release es una versión publicada de un proyecto."),
    ("lock", "Un lock evita que ciertas operaciones incompatibles ocurran al mismo tiempo."),
    ("buffer", "Un buffer guarda temporalmente datos mientras se transfieren o procesan."),
    ("stream", "Un stream representa una secuencia de datos que se procesa progresivamente."),
    ("worker", "Un worker ejecuta tareas asignadas dentro de un sistema."),
    ("evento", "Un evento representa algo que ocurrió y que otro componente puede atender."),
    ("callback", "Un callback es una función que se ejecuta cuando ocurre una condición prevista."),
]


TRAIN_TASKS = [
    ("una aplicacion no abre", "Revisá el mensaje de error o registro disponible.", "Comprobá la configuración y dependencias necesarias."),
    ("una pagina no carga", "Comprobá que la conexión de red funcione.", "Revisá si el navegador o el sitio informan un error."),
    ("un programa se cierra solo", "Revisá el error o registro generado al cerrarse.", "Comprobá dependencias y configuración del programa."),
    ("un dispositivo usb no aparece", "Comprobá la conexión física y probá otro puerto.", "Revisá si el sistema detecta el dispositivo o informa un error."),
    ("un servicio no inicia", "Revisá el mensaje o registro de error del servicio.", "Comprobá su configuración y dependencias requeridas."),
    ("un comando falla", "Leé el mensaje de error que devuelve el comando.", "Verificá parámetros, rutas y permisos usados."),
    ("una conexion se corta", "Comprobá si la red local sigue disponible.", "Revisá el servicio remoto y cualquier error registrado."),
    ("un proceso consume demasiado", "Identificá el proceso y medí su consumo actual.", "Revisá qué tarea realiza y si registra errores."),
    ("un modulo no carga", "Revisá el error exacto al cargar el módulo.", "Comprobá que la dependencia esté instalada y sea compatible."),
    ("un puerto no responde", "Comprobá si el servicio asociado está iniciado.", "Revisá conectividad y configuración del puerto."),
    ("un login falla", "Revisá el mensaje de error del acceso.", "Comprobá credenciales, permisos y estado del servicio."),
    ("una request devuelve error", "Revisá el código y mensaje de respuesta.", "Comprobá parámetros, destino y autenticación usados."),
    ("un worker deja tareas pendientes", "Revisá el estado y los logs del worker.", "Comprobá la cola y los errores de las tareas."),
    ("una dependencia rompe al actualizar", "Identificá la versión que introdujo el error.", "Revisá compatibilidad y cambios de configuración requeridos."),
    ("un backup no termina", "Revisá el error y el espacio disponible.", "Comprobá conectividad y destino del backup."),
    ("un proxy no conecta", "Comprobá que el proxy esté disponible.", "Revisá host, puerto y credenciales configuradas."),
    ("un endpoint responde lento", "Medí el tiempo y revisá los logs del endpoint.", "Comprobá carga, red y dependencias externas."),
    ("un proceso queda bloqueado", "Identificá dónde deja de avanzar.", "Revisá locks, recursos y operaciones pendientes."),
]

VALID_TASKS = [
    ("un daemon no arranca", "Revisá el mensaje o log de inicio.", "Comprobá configuración y dependencias del daemon."),
    ("una release falla al iniciar", "Revisá el error exacto de la nueva versión.", "Compará configuración y dependencias con la versión anterior."),
    ("un stream se interrumpe", "Revisá dónde termina la transferencia.", "Comprobá red, origen y destino del stream."),
    ("un callback no se ejecuta", "Confirmá que la condición esperada ocurra.", "Revisá el registro y asociación del callback."),
    ("un buffer se llena", "Medí cuánto tarda en consumirse.", "Revisá productor, consumidor y límites configurados."),
    ("una cola crece sin parar", "Comprobá si los workers están procesando tareas.", "Revisá errores y velocidad de entrada y salida."),
]


TRAIN_MISSING = [
    ("quiero convertir algo pero no te dije a que formato", "el formato de destino"),
    ("quiero abrir una ubicacion pero no te dije cual", "qué ubicación"),
    ("quiero calcular un total pero no te pase los valores", "los valores necesarios"),
    ("quiero mandar algo pero no dije a quien", "el destinatario"),
    ("quiero buscar algo pero no dije que cosa", "qué querés buscar"),
    ("quiero mover un elemento pero no te dije donde", "el destino"),
    ("quiero renombrar algo pero no te dije el nuevo nombre", "el nombre nuevo"),
    ("quiero conectarme a un servidor pero no dije cual", "qué servidor"),
    ("quiero filtrar una lista pero no te di el criterio", "el criterio de filtrado"),
    ("quiero comparar resultados pero solo te pase uno", "el segundo resultado"),
    ("quiero programar una tarea pero no dije cuando", "la fecha u horario"),
    ("quiero hacer una request pero no dije el endpoint", "el endpoint"),
    ("quiero autenticarme pero no indique la cuenta", "la cuenta"),
    ("quiero restaurar un backup pero no dije cual", "qué backup"),
    ("quiero revisar un log pero no indique de que proceso", "el proceso o servicio"),
    ("quiero cambiar una configuracion pero no dije el valor nuevo", "el valor nuevo"),
]

VALID_MISSING = [
    ("quiero crear una tarea pero no dije fecha ni hora", "la fecha y la hora"),
    ("quiero copiar datos pero no dije la ubicacion final", "el destino"),
    ("quiero comparar dos resultados pero solo te indique uno", "el segundo resultado"),
    ("quiero enviar un mensaje pero no dije el contacto", "el destinatario"),
    ("quiero buscar por fecha pero no indique ninguna", "la fecha o rango"),
    ("quiero llamar un servicio pero no dije la operacion", "la operación"),
    ("quiero conectarme pero no indique host ni puerto", "el host y el puerto"),
    ("quiero desplegar una release pero no dije cual", "qué release"),
]


SHORT_ANSWERS = {
    "mouse": "Mueve el puntero y permite seleccionar elementos.",
    "teclado": "Permite escribir texto y ejecutar comandos.",
    "monitor": "Muestra visualmente información de la computadora.",
    "backup": "Permite recuperar datos cuando se pierden.",
    "navegador": "Permite abrir y recorrer sitios web.",
    "variable": "Guarda un valor que puede modificarse.",
    "base de datos": "Organiza datos para guardarlos y consultarlos.",
    "contraseña": "Protege el acceso a una cuenta.",
    "actualizacion": "Corrige errores o incorpora mejoras al software.",
    "proceso": "Representa una tarea o programa en ejecución.",
    "registro": "Guarda eventos para revisarlos después.",
    "puerto": "Identifica un punto de comunicación de un servicio.",
    "cache": "Guarda datos temporales para reutilizarlos rápido.",
    "ruta": "Indica la ubicación de un elemento.",
    "servidor": "Ofrece datos o servicios a otros programas.",
    "cliente": "Solicita datos o funciones a otro sistema.",
    "dependencia": "Es un componente requerido por otro programa.",
    "configuracion": "Controla mediante valores cómo funciona un programa.",
    "extension": "Ayuda a identificar el tipo de un elemento.",
    "directorio": "Organiza elementos dentro del sistema.",
    "cola": "Organiza elementos para atenderlos en orden.",
    "indice": "Facilita búsquedas mediante referencias organizadas.",
    "sesion": "Representa un período de interacción.",
    "variable de entorno": "Guarda un valor disponible para procesos.",
    "paquete": "Agrupa componentes que se distribuyen juntos.",
    "servicio": "Ofrece una función ejecutándose en segundo plano.",
    "acceso directo": "Permite abrir rápidamente otro elemento.",
    "log": "Registra eventos para revisar lo ocurrido.",
    "checksum": "Permite detectar cambios en datos.",
    "branch": "Separa una línea de cambios del proyecto.",
    "commit": "Registra un conjunto concreto de cambios.",
    "socket": "Representa un punto de comunicación.",
    "request": "Es una solicitud enviada a otro servicio.",
    "response": "Es la respuesta devuelta a una solicitud.",
    "cache local": "Conserva resultados temporales para reutilizarlos.",
    "permiso": "Define qué acciones están autorizadas.",
    "daemon": "Es un proceso ejecutado en segundo plano.",
    "proxy": "Intermedia entre una solicitud y su destino.",
    "endpoint": "Identifica un punto donde se ofrece una función.",
    "repositorio": "Organiza versiones de un proyecto.",
    "parche": "Agrupa cambios para corregir o modificar software.",
    "release": "Es una versión publicada de un proyecto.",
    "lock": "Evita operaciones incompatibles al mismo tiempo.",
    "buffer": "Guarda datos temporalmente durante su procesamiento.",
    "stream": "Representa una secuencia progresiva de datos.",
    "worker": "Ejecuta tareas asignadas dentro de un sistema.",
    "evento": "Representa algo ocurrido que puede atenderse.",
    "callback": "Ejecuta una función ante una condición prevista.",
}


def _rows_one_sentence(facts, templates, split):
    rows = []
    for i, (topic, answer) in enumerate(facts):
        for j, template in enumerate(templates):
            rows.append(row(
                template.format(topic=topic),
                answer,
                f"instruction:sentence:{split}:{i}:{j}",
                "instruction_following",
            ))
    return rows


def _rows_two_steps(tasks, templates, split):
    rows = []
    for i, (task, first, second) in enumerate(tasks):
        answer = f"1. {first}\n2. {second}"
        for j, template in enumerate(templates):
            rows.append(row(
                template.format(task=task),
                answer,
                f"instruction:two:{split}:{i}:{j}",
                "instruction_following",
            ))
    return rows


def _rows_short(facts, templates, split):
    rows = []
    for i, (topic, _) in enumerate(facts):
        answer = SHORT_ANSWERS[topic]
        for j, template in enumerate(templates):
            rows.append(row(
                template.format(topic=topic),
                answer,
                f"instruction:short:{split}:{i}:{j}",
                "instruction_following",
            ))
    return rows


def _rows_missing(cases, templates, split):
    rows = []
    for i, (case, missing) in enumerate(cases):
        answer = (
            f"Me falta {missing}. Necesito ese dato antes de continuar; "
            "no debería inventarlo ni asumirlo."
        )
        for j, template in enumerate(templates):
            rows.append(row(
                template.format(case=case),
                answer,
                f"instruction:missing:{split}:{i}:{j}",
                "instruction_following",
            ))
    return rows


def _augment_train(rows, rng):
    prefixes = ["porfa ", "che ", "por favor ", "necesito que ", "sin vueltas "]
    out = list(rows)
    for item in rows:
        for prefix in rng.sample(prefixes, 2):
            clone = dict(item)
            clone["user"] = prefix + clone["user"]
            clone["family"] = clone["family"] + ":aug:" + prefix.strip().replace(" ", "_")
            out.append(clone)
    return out


def build(seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)

    train = []
    train += _rows_one_sentence(
        TRAIN_FACTS,
        [
            "en una sola oracion explicame que es {topic}",
            "solo una oracion para definir {topic}",
            "decime que es {topic} usando exactamente una oracion",
        ],
        "train",
    )
    train += _rows_two_steps(
        TRAIN_TASKS,
        [
            "dame exactamente dos pasos para revisar por que {task}",
            "solo dos pasos numerados para diagnosticar por que {task}",
            "responde con dos pasos y nada mas para revisar por que {task}",
        ],
        "train",
    )
    train += _rows_short(
        TRAIN_FACTS,
        [
            "en menos de 10 palabras para que sirve {topic}",
            "menos de diez palabras: utilidad de {topic}",
            "resumilo en menos de 10 palabras: {topic}",
        ],
        "train",
    )
    train += _rows_missing(
        TRAIN_MISSING,
        [
            "{case}; que haces sin inventar",
            "{case}, que dato pedirias antes de seguir",
            "{case}; responde que informacion falta",
        ],
        "train",
    )
    train = _augment_train(train, rng)

    valid = []
    valid += _rows_one_sentence(
        VALID_FACTS,
        [
            "explicame {topic} pero en exactamente una oracion",
            "una unica oracion para decir que es {topic}",
            "sin lista y en una sola oracion explica {topic}",
        ],
        "valid",
    )
    valid += _rows_two_steps(
        VALID_TASKS,
        [
            "exactamente dos pasos para comprobar por que {task}",
            "sin agregar un tercer paso: dos pasos para revisar por que {task}",
            "dos pasos numerados y nada mas para diagnosticar por que {task}",
        ],
        "valid",
    )
    valid += _rows_short(
        VALID_FACTS,
        [
            "menos de diez palabras para explicar la utilidad de {topic}",
            "en menos de 10 palabras decime para que sirve {topic}",
            "resumi {topic} en menos de diez palabras",
        ],
        "valid",
    )
    valid += _rows_missing(
        VALID_MISSING,
        [
            "{case}; que deberias pedir antes de continuar",
            "{case}, que informacion falta",
            "{case}; no inventes: indica que dato necesitas",
        ],
        "valid",
    )

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid
