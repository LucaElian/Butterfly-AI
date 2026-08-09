# ButterflyAI v0.0003 — instalación permanente

Desde esta versión dejamos de crear una carpeta nueva por cada build. Esta carpeta se llama simplemente `ButterflyAI` y se actualiza en el futuro.

La política de continuidad es:

1. Se heredan datasets, memoria y conocimiento útil de la Butterfly anterior.
2. La Butterfly anterior también genera una pequeña colección de respuestas; el profesor local las corrige antes de convertirlas en material heredado.
3. Se genera nuevo material de estudio con mucho más peso en conversación cotidiana y lenguaje.
4. Se entrena ButterflyTokenizer v2, un tokenizer propio que combina fallback UTF-8 con palabras/subpalabras aprendidas.
5. Se entrena un Transformer nuevo más grande para esta generación.
6. Las experiencias futuras se aprenden mediante `SLEEP_AND_LEARN.bat`.
7. Una candidata solo reemplaza a la activa si mejora la evaluación. Después de promocionarla, el checkpoint anterior se elimina para ahorrar espacio. Los datos/memoria no se borran.

## Hardware objetivo

Esta build está ajustada para:

- AMD Ryzen 5 3600 (6C/12T)
- 16 GB RAM
- Radeon RX 580 8 GB

El entrenamiento principal usa CPU en Windows por estabilidad. El perfil por defecto ronda ~16M parámetros (el número exacto depende del vocabulario aprendido).

## Orden para pasar desde v0.0002

### 0. Descomprimir

Descomprimí esta versión en una carpeta nueva llamada, por ejemplo:

`D:\ButterflyAI`

A partir de ahora esa será la carpeta permanente.

### 1. SETUP_WINDOWS.bat

Crea `.venv`, instala dependencias e inicializa memoria.

### 2. 01_MIGRATE_PREVIOUS_AND_CLEAN.bat

Arrastrá la carpeta vieja `ButterflyAI-v0.0002` a la consola cuando te la pida.

El migrador:

- importa los `.jsonl` de entrenamiento anteriores;
- combina la memoria SQLite;
- carga el último modelo anterior;
- le hace responder un pequeño examen de herencia;
- un profesor local corrige esas respuestas para no copiar balbuceos/errores;
- guarda esas lecciones corregidas en la nueva Butterfly;
- escribe un informe de migración;
- **solo al final** pregunta si querés borrar la carpeta anterior.

Para eliminarla tenés que escribir exactamente `BORRAR`. Si algo falla antes, el script no llega a esa fase.

> Nota: el cache descargado del profesor Qwen vive normalmente en el cache de Hugging Face de Windows y se reutiliza; no se duplica dentro de cada Butterfly.

### 3. 02_BUILD_CONSOLIDATED_DATASET.bat

Crea `data/consolidated.*` combinando:

- material heredado;
- lecciones corregidas de la Butterfly anterior;
- identidad estable de ButterflyAI;
- cientos de ejemplos nuevos del profesor local;
- más saludos, conversación, lenguaje cotidiano, explicaciones, razonamiento, epistemología, programación y tareas de PC.

Los saludos/conversación tienen ahora mucho más peso para evitar el problema de que Butterfly responda a `Hola` con un plan para ordenar archivos.

### 4. 03_TRAIN_NEW_TOKENIZER.bat

Entrena `ButterflyTokenizer v2` usando el corpus consolidado.

A diferencia del tokenizer byte-a-byte anterior, aprende piezas frecuentes como palabras y subpalabras, pero conserva fallback por bytes para nunca quedar sin representación de un carácter.

### 5. 04_TRAIN_BUTTERFLY.bat

Entrena `ButterflyAI v0.0003` con la arquitectura nueva y el tokenizer nuevo.

Como la arquitectura/tokenizer cambian, los pesos de v0.0002 no se pueden copiar directamente. La continuidad se mantiene mediante los datasets, memoria y las lecciones heredadas.

### 6. 05_EVALUATE_BUTTERFLY.bat

Mide pérdida de validación + regresiones epistemológicas y muestra respuestas de prueba a:

- Hola
- Buenas
- Que estas diciendo?
- Que haces si no sabes algo?
- Explica que es un archivo.

### 7. START_CHAT.bat

Abre el chat local.

## Aprendizaje posterior

`SLEEP_AND_LEARN.bat` usa únicamente experiencias verificadas y de buena calidad para crear una candidata.

Si mejora la evaluación:

- se promociona;
- se conserva toda la memoria/dataset;
- se elimina el checkpoint anterior.

Si empeora:

- se rechaza;
- se elimina la candidata;
- la Butterfly estable sigue intacta.

Así se aplica la idea del "libro": el contenido útil pasa al libro actual; los checkpoints viejos se queman cuando ya no hacen falta.

## Archivos que representan la vida de Butterfly

No deberían borrarse durante futuras actualizaciones:

- `.butterfly/butterfly.db`
- `.butterfly/tokenizer-v2.json` (mientras siga siendo compatible)
- `data/inherited/`
- `data/consolidated.jsonl`
- experiencias verificadas de la base de datos
- reglas/conocimiento verificado

Los modelos `.pt` sí pueden compactarse a uno solo después de cada promoción exitosa.
