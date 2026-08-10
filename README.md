# ButterflyAI

ButterflyAI usa **una sola instalación permanente**. El código, la memoria, el corpus,
los benchmarks y el historial evolucionan en el mismo proyecto. Los nombres de los
scripts del pipeline ya no cambian con cada generación.

## Estado actual tras la limpieza de infraestructura

- cerebro aceptado: **v0.0004**;
- arquitectura: **17,477,376 parámetros**;
- tokenizer aceptado: **v3 / 8,192 tokens**;
- benchmark estricto actual: **v0.00042**;
- v0.0005: candidata rechazada;
- v0.00051: candidata rechazada;
- próxima generación deliberada: **v0.00052**;
- `config/pipeline.json` queda sin target hasta instalar la receta v0.00052.

Las candidatas rechazadas no se conservan físicamente. Corpus, memoria, conocimiento
verificado, manifests, benchmarks e historial sí se conservan.

## Pipeline permanente

Estos nombres quedan fijos:

```text
01_PREPARE.bat
02_BUILD_DATASET.bat
03_TRAIN.bat
04_EVALUATE_AND_PROMOTE.bat
RUN_PIPELINE.bat
```

`RUN_PIPELINE.bat` permite:

1. automático;
2. pausado entre etapas;
3. reanudar desde la primera etapa incompleta;
4. ejecutar una sola etapa.

Si una etapa falla, el modo automático se detiene y no ejecuta las posteriores.

## Logs

Cada ejecución crea:

```text
logs\pipeline-AAAA-MM-DD_HH-MM-SS.log
logs\latest.log
logs\latest-error.log        (solo al fallar)
reports\latest-summary.txt
```

La salida sigue apareciendo en la consola y simultáneamente se guarda en el log.

## Filosofía de almacenamiento

- un solo cerebro físico activo;
- candidata física solo mientras está siendo evaluada;
- old brain se elimina **solo después** de una promoción verificada;
- modelos estables usan `safetensors` weights-only;
- optimizer/recovery state es temporal y queda fuera de Git;
- memoria local (`.butterfly/butterfly.db`) queda fuera de Git;
- modelos físicos quedan fuera del historial Git normal.

## Git

Antes de commit revisá:

```text
git status
```

`models/history.json`, código, tokenizer, manifests y benchmarks pueden formar parte del
historial. El cerebro físico, DB local, logs, reports y training state no.

## Utilidades permanentes

```text
SETUP_WINDOWS.bat
START_CHAT.bat
STATUS.bat
STORAGE_STATUS.bat
EVALUATE_ACTIVE.bat
VERIFY_EXAMPLE.bat
SLEEP_AND_LEARN.bat
EXPORT_ACTIVE_MODEL_FOR_RELEASE.bat
```

Sleep learning queda pausado por `config/pipeline.json` durante la transición deliberada
a v0.00052, para que no cree una candidata paralela accidentalmente.

## Versiones

No existe un archivo raíz `VERSION` que haya que renombrar en cada experimento.

- el cerebro activo sale del registry local;
- el target deliberado sale de `config/pipeline.json`;
- la suite de evaluación sale del evaluador;
- el historial de experimentos vive en benchmarks + `models/history.json` + `CHANGELOG.md`.

Así evitamos tener cuatro fuentes distintas diciendo versiones diferentes.
