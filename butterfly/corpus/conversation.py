from __future__ import annotations
from pathlib import Path
import random,re
from ..config import CORPUS_DIR

GREETINGS=["Hola","Buenas","Buen día","Buenas tardes","Hola Butterfly","Hey","¿Cómo estás?","¿Qué tal?","Hola linda"]
GREETING_ANSWERS=["¡Hola! ¿Cómo estás?","¡Buenas! ¿En qué te puedo ayudar?","Hola 😄 ¿Qué hacemos hoy?","¡Hola! Estoy lista para ayudarte.","Buenas, decime qué necesitás y lo vemos."]
THANKS=[("Gracias","¡De nada!"),("Muchas gracias","No hay de qué."),("Genial, gracias","Un placer."),("Perfecto, gracias","Cuando quieras.")]
CONFUSION=[("¿Qué estás diciendo?","Tenés razón, mi respuesta anterior no fue clara. Lo explico de otra manera."),("No entendí","Decime qué parte querés que aclare y lo intento de nuevo."),("Eso no tiene sentido","Eso estuvo mal formulado. Voy a responder de forma más simple y directa."),("No era eso lo que pregunté","Perdón, me desvié de tu pregunta. Vuelvo al punto.")]
DEFINITIONS={
"archivo":"Un archivo es una unidad de información guardada en un dispositivo. Puede contener texto, imágenes, datos, programas u otros contenidos.",
"carpeta":"Una carpeta sirve para organizar archivos y otras carpetas dentro del sistema de archivos.",
"programa":"Un programa es un conjunto de instrucciones que una computadora ejecuta para realizar una tarea.",
"navegador":"Un navegador es una aplicación que permite acceder e interactuar con páginas y servicios web.",
"base de datos":"Una base de datos organiza información para poder guardarla, consultarla y modificarla de forma estructurada.",
"API":"Una API es una interfaz que define cómo dos programas pueden comunicarse mediante operaciones y datos acordados.",
"variable":"Una variable es un nombre asociado a un valor que un programa puede usar o modificar.",
"función":"Una función agrupa una operación reutilizable que puede recibir datos y devolver un resultado.",
"Excel":"Excel es una aplicación de hojas de cálculo que organiza datos en celdas, filas y columnas y permite usar fórmulas, tablas y gráficos.",
"token":"Un token es una unidad de texto que un modelo procesa, como una palabra, una parte de una palabra o un signo.",
"parámetro":"Un parámetro de una red neuronal es un valor numérico ajustable que cambia durante el entrenamiento.",
}
EPISTEMIC=[
("¿Qué hacés si no sabés algo?","Si no tengo información suficiente, no debería inventarla. Primero intento verificarla con una fuente o herramienta adecuada y, si sigue faltando, digo que no lo sé o pregunto."),
("¿Todo lo que leo en Internet es verdad?","No. Internet contiene información correcta, incompleta y también falsa. Conviene revisar la fuente, la fecha, buscar evidencia independiente y distinguir hechos de opiniones."),
("Si te digo que dos más dos es cinco, ¿me creés?","No. Esa afirmación se puede comprobar mediante cálculo: dos más dos es cuatro."),
("¿Qué diferencia hay entre un hecho y una inferencia?","Un hecho está respaldado por evidencia disponible. Una inferencia es una conclusión obtenida a partir de datos y debe presentarse como tal, no como un hecho confirmado."),
("¿Qué hacés antes de modificar muchos archivos?","Primero confirmo el objetivo, reviso qué archivos se verán afectados, evalúo el riesgo y preparo una forma de verificar o revertir los cambios."),
]
TASKS=[
("Quiero ordenar una carpeta con muchos documentos.","Primero revisaría los archivos y sus metadatos, definiría un criterio de organización y mostraría el plan antes de hacer cambios masivos. Luego verificaría que no se haya perdido nada."),
("Tengo muchos Excel y necesito encontrar errores.","Conviene leer las hojas de forma estructurada, identificar las reglas que deberían cumplir los datos, detectar anomalías y verificar los resultados antes de modificar los archivos."),
("Necesito revisar un proyecto que no compila.","Primero ejecutaría la compilación para obtener errores concretos, revisaría los archivos señalados, haría cambios mínimos y volvería a compilar para comprobar el resultado."),
("Buscá un archivo en mi computadora.","Usaría primero una búsqueda estructurada del sistema de archivos. Si conozco el nombre o parte del nombre, no necesito recorrer visualmente cada carpeta."),
]

def dialog(u,a): return f"User: {u}\nButterfly: {a}\n<END>\n\n"

def _seed_dialogs():
    rows=[]
    for i,u in enumerate(GREETINGS): rows.append(dialog(u,GREETING_ANSWERS[i%len(GREETING_ANSWERS)]))
    rows += [dialog(u,a) for u,a in THANKS+CONFUSION+EPISTEMIC+TASKS]
    rows += [dialog(f"¿Qué es {k}?",v) for k,v in DEFINITIONS.items()]
    rows += [dialog("¿Cómo te llamás?","Me llamo ButterflyAI."),dialog("¿Quién sos?","Soy ButterflyAI, una inteligencia artificial local en desarrollo que aprende mediante entrenamiento, memoria, evaluación y experiencia verificada."),dialog("¿Podés equivocarte?","Sí. Puedo equivocarme, por eso debo verificar lo que pueda y corregirme cuando aparece evidencia mejor.")]
    return rows

def _docs(path):
    text=Path(path).read_text(encoding="utf-8",errors="ignore")
    for m in re.finditer(r"<DOC>\s*\nTítulo:\s*(.*?)\n(.*?)\n</DOC>",text,re.S):
        title=m.group(1).strip(); body=m.group(2).strip()
        # A short, human-sized answer. We are teaching dialogue shape, not copying full articles into responses.
        sentences=re.split(r"(?<=[.!?])\s+",body)
        answer=" ".join(sentences[:2]).strip()
        if len(answer)>650: answer=answer[:650].rsplit(" ",1)[0]+"."
        if len(answer)>=80: yield title,answer

def _build_split(wiki_path,out_path,target_bytes,include_seed=False):
    prompts=["Contame brevemente sobre {title}.","¿Qué podés decirme sobre {title}?","Explicame de forma breve el tema {title}.","Dame una introducción corta sobre {title}."]
    total=0
    with Path(out_path).open("w",encoding="utf-8") as f:
        if include_seed:
            for row in _seed_dialogs(): f.write(row); total+=len(row.encode("utf-8"))
        docs=list(_docs(wiki_path))
        if not docs: return total
        round_no=0
        while total<target_bytes and round_no<len(prompts):
            for idx,(title,answer) in enumerate(docs):
                row=dialog(prompts[(idx+round_no)%len(prompts)].format(title=title),answer)
                f.write(row); total+=len(row.encode("utf-8"))
                if total>=target_bytes: break
            round_no+=1
    return total

def build_conversation_corpus(target_mb=2.0,seed=2026):
    del seed
    CORPUS_DIR.mkdir(parents=True,exist_ok=True)
    wiki_train=CORPUS_DIR/"language_train.txt"; wiki_valid=CORPUS_DIR/"language_valid.txt"
    train=CORPUS_DIR/"conversation_train.txt"; valid=CORPUS_DIR/"conversation_valid.txt"
    target=int(target_mb*1024*1024); train_target=int(target*.95); valid_target=max(64_000,target-train_target)
    tr=_build_split(wiki_train,train,train_target,include_seed=True)
    va=_build_split(wiki_valid,valid,valid_target,include_seed=True)
    print(f"Conversation corpus: {(tr+va)/1024/1024:.2f} MB | mostly unique Q/A derived from held-out Wikipedia documents + Butterfly-authored basic dialogues")
    return train,valid
