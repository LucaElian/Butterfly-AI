# ButterflyAI v0.0004 — actualización definitiva

ButterflyAI v0.0004 actualiza **la instalación permanente** de ButterflyAI. No crea una vida nueva y no requiere borrar v0.0003 antes de empezar.

## Objetivo de v0.0004

v0.0003 demostró que el Transformer funciona, pero tenía ~15.9M parámetros y solo ~53k tokens de educación. El resultado fue overfitting: aprendía estructuras del pequeño dataset pero hablaba mal fuera de él.

v0.0004 mantiene una escala parecida de modelo y cambia el cuello de botella:

- corpus lingüístico de varios millones de tokens;
- español general antes de conversación;
- conversación antes de instrucciones/tareas;
- tokenizer subword v3 de 8,192 tokens con fallback UTF-8;
- validación por etapa + early stopping;
- candidata obligatoriamente comparada contra v0.0003;
- promoción solo si mejora de forma medible;
- memoria/datos útiles heredados;
- respuestas defectuosas de v0.0003 NO se usan como lenguaje maestro;
- modelo estable guardado como **safetensors weights-only**, sin estado de Adam.

## INSTALACIÓN SOBRE TU BUTTERFLY ACTUAL

Tu carpeta permanente debería ser algo como:

    D:\Downloads\ButterflyAI\

1. Hacé un commit/tag de v0.0003 si querés conservar el snapshot histórico en GitHub.
2. Extraé este ZIP **dentro de esa misma carpeta ButterflyAI**.
3. Permití reemplazar archivos con el mismo nombre.
4. NO borres manualmente `.butterfly`, `models`, `data` ni el modelo v0.0003.
5. Ejecutá `SETUP_WINDOWS.bat`.

El updater no incluye tu DB, tu checkpoint ni tu corpus local, por lo que no los pisa al extraerlo.

## ORDEN DE EJECUCIÓN

### 1. SETUP_WINDOWS.bat
Actualiza dependencias y elimina scripts/código obsoleto de la pipeline v0.0003, pero no borra la vida de Butterfly.

### 2. 00_PREPARE_V0004.bat
- confirma que existe una Butterfly activa;
- conserva una copia identificable del tokenizer v0.0003;
- evalúa v0.0003 con el examen nuevo;
- guarda ese benchmark como baseline;
- si el cerebro activo todavía es el `.pt` grande de v0.0003, lo convierte a un `.safetensors` weights-only, recarga y compara TODOS los pesos; solo si son idénticos elimina el `.pt` local con el estado de Adam. Esto no cambia la inteligencia ni la versión, solo compacta almacenamiento.

### 3. 01_BUILD_LANGUAGE_CORPUS.bat
Construye el nuevo libro de estudio:

- ~20 MB de español general desde Wikipedia ES;
- ~2 MB de conversación;
- instrucciones/epistemología/datos heredados útiles.

La descarga de Wikipedia es reanudable. Si se corta, ejecutá el BAT otra vez.

Archivos grandes generados quedan fuera de Git mediante `.gitignore`.

### 4. 02_TRAIN_TOKENIZER_V3.bat
Entrena el tokenizer propio de Butterfly:

- vocabulario objetivo: 8,192;
- whole words + subwords;
- fallback UTF-8 para texto nunca visto;
- trie para encoding rápido.

### 5. 03_TRAIN_BUTTERFLY_V0004.bat
Crea una candidata con pesos nuevos, porque cambió el significado del vocabulario/embeddings.

Curriculum:

    LANGUAGE
      -> CONVERSATION
      -> INSTRUCTION

Cada etapa usa validación. Conversation e instruction usan early stopping y restauran el mejor epoch en vez de guardar ciegamente el último.

La candidata todavía NO reemplaza a v0.0003.

### 6. 04_COMPARE_AND_PROMOTE.bat
Ejecuta el mismo examen sobre v0.0003 y v0.0004.

Si v0.0004 gana por el margen exigido:

- pasa a activa;
- se renombra como cerebro estable;
- queda guardada como `.safetensors` **solo con pesos de inferencia**;
- se registra en `models/history.json`;
- recién entonces se borra el checkpoint físico v0.0003 y su tokenizer obsoleto;
- se conservan memoria, experiencias, corpus, fuentes, reglas y benchmarks.

Si v0.0004 pierde:

- se borra solo la candidata;
- v0.0003 sigue activa;
- corpus/tokenizer nuevo se conservan para reintentar sin repetir el trabajo caro.

### 7. START_CHAT.bat
Habla con la Butterfly que haya quedado activa.

---

# Nuevo sistema de almacenamiento

## Por qué v0.0003 pesaba ~190 MB

El checkpoint v0.0003 guardaba:

- pesos del modelo (~64 MB en float32);
- primer momento de Adam (~64 MB);
- segundo momento de Adam (~64 MB);
- metadata.

Por eso un modelo de ~15.9M parámetros terminaba cerca de 190 MB.

## Desde v0.0004

El cerebro aceptado se guarda como:

    models/butterfly-v0.0004.safetensors
    models/butterfly-v0.0004.safetensors.json

Solo contiene los pesos necesarios para inferencia + metadata pequeña. No contiene optimizer.

Con un modelo de tamaño parecido a v0.0003 debería estar mucho más cerca de ~60-70 MB que de ~190 MB.

## GitHub

Los cerebros físicos, DB local, training state, release bundles y corpus masivos están ignorados por `.gitignore`.

El repo debería guardar principalmente:

- código;
- scripts;
- configuraciones;
- seeds pequeños;
- benchmarks;
- `models/history.json`;
- manifests compactos;
- tokenizer activo (es pequeño y necesario para reproducibilidad).

El cerebro se distribuye como **GitHub Release asset**, no como archivo normal del repo.

Después de promocionar v0.0004 ejecutá:

    EXPORT_ACTIVE_MODEL_FOR_RELEASE.bat

Genera en `release\`:

    ButterflyAI-brain-v0.0004.zip

que incluye:

- `.safetensors` weights-only;
- metadata/config;
- tokenizer;
- SHA256 manifest.

Subí ese ZIP como asset de un Release de GitHub.

### Si el repo ya rastrea runtime/modelos viejos

Podés ejecutar opcionalmente:

    GITHUB_STOP_TRACKING_RUNTIME.bat

Pide confirmación explícita. Usa `git rm --cached`: **deja de rastrear archivos en Git sin borrarlos de tu disco**. Revisá `git status` antes de commit/push.

El tag/commit v0.0003 seguirá conservando históricamente el snapshot viejo.

---

# Archivos útiles extra

- `STATUS.bat`: estado lógico de Butterfly.
- `STORAGE_STATUS.bat`: historial y ubicación del cerebro activo.
- `EVALUATE_ACTIVE.bat`: examen de la Butterfly activa.
- `VERIFY_EXAMPLE.bat`: prueba del EpistemicEngine.
- `SLEEP_AND_LEARN.bat`: aprendizaje con experiencias verificadas + memory replay.
- `EXPORT_ACTIVE_MODEL_FOR_RELEASE.bat`: paquete del cerebro para GitHub Release.
- `OPTIONAL_DELETE_QWEN_TEACHER_CACHE.bat`: elimina el viejo cache de Qwen solo si ya no lo necesitás.

# Regla permanente de evolución

Una nueva Butterfly **no se vuelve estable porque terminó de entrenar**.

Se vuelve estable únicamente si:

1. termina el entrenamiento;
2. pasa la evaluación;
3. supera a la Butterfly activa;
4. no regresa en criterios críticos;
5. recién entonces reemplaza el cerebro anterior.

Los checkpoints viejos/rechazados se queman para ahorrar espacio; el conocimiento útil consolidado, memoria, experiencias verificadas y métricas históricas continúan.
