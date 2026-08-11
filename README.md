# ButterflyAI

ButterflyAI usa una sola instalación permanente. Las versiones de cerebro y la identidad
del evaluador son **datos de runtime/historial**, no constantes escritas a mano en el código.

## Estados de modelo

Butterfly mantiene como máximo tres slots físicos:

- **ACTIVE**: cerebro aceptado y seguro para uso normal.
- **LAB**: mejor cerebro experimental acumulado; todavía no reemplaza a ACTIVE.
- **CANDIDATE**: experimento que está entrenándose o evaluándose.

Una candidata puede pasar a LAB si mejora las métricas foco de su receta sin romper las
capacidades protegidas. Solo pasa a ACTIVE si además cumple la política global y todos los
hard gates del evaluador.

## Pipeline permanente

```text
01_PREPARE.bat
02_BUILD_DATASET.bat
03_TRAIN.bat
04_EVALUATE_AND_PROMOTE.bat
RUN_PIPELINE.bat
```

`RUN_PIPELINE.bat` ofrece:

1. **AUTOMATICO**: busca la primera etapa pendiente y continúa hasta el final.
2. **PAUSADO**: igual que automático, pero espera ENTER entre etapas.
3. **UNA ETAPA**: ejecuta solo la etapa elegida y deja el estado listo para continuar.

No existe un modo `REANUDAR` separado: reanudar es el comportamiento normal.

## Estado vs configuración

`config/pipeline.json` describe comportamiento permanente. No guarda target ni suite.

El experimento actual vive localmente en:

```text
.butterfly/current_experiment.json
```

El registry local mantiene ACTIVE/LAB/CANDIDATE:

```text
models/registry.json
```

Las recetas reutilizables viven en:

```text
config/recipes/
```

Una receta define objetivo, skills, hiperparámetros, métricas foco y métricas protegidas.
No contiene la versión del cerebro ni la versión del evaluador.

## Evaluador

La suite no tiene un número manual que haya que incrementar. Su ID se calcula
automáticamente a partir de casos, thresholds, valores reservados y reglas semánticas.
Si cambia el examen, cambia su fingerprint.

## Logs

Cada etapa escribe un log descriptivo independiente:

```text
AAAA-MM-DD_HH-MM-SS__prepare__target-vX__recipe-NOMBRE.log
AAAA-MM-DD_HH-MM-SS__build-dataset__target-vX__recipe-NOMBRE.log
AAAA-MM-DD_HH-MM-SS__train__target-vX__recipe-NOMBRE.log
AAAA-MM-DD_HH-MM-SS__evaluate-and-promote__target-vX__recipe-NOMBRE.log
```

`latest.log` conserva la salida completa de las etapas ejecutadas en la invocación actual.

También se mantienen:

```text
logs/latest.log
logs/latest-error.log
reports/latest-summary.txt
```

## Auditoría de hardcodes

```text
python -m butterfly audit-hardcodes
```

La auditoría falla si encuentra versiones de cerebro/suite escritas en código operativo,
constantes de tokenizer ligadas a generaciones o rutas absolutas de ButterflyAI.

## Git

Se versionan código, configuración reusable, tests, manifests pequeños, benchmarks,
CHANGELOG e historial. No se versionan pesos físicos, DB personal, recovery state, logs
ni el experimento runtime actual.
