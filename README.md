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

## Night Study

`RUN_NIGHT_STUDY.bat` ejecuta el estudio autonomo con los limites definidos en
`config/night_study.json`. En modo Lifelong puede continuar hasta que se pida una parada
segura con `STOP_NIGHT_STUDY.bat`.

`LIFELONG_STATUS.bat` muestra el estado del curriculum graph y hace un chequeo rapido del
motor de examenes dinamicos.

`CRASH_RECOVERY_STATUS.bat` muestra si hay una corrida interrumpida que Night Study puede
reanudar, y `SKILL_CREDIT_STATUS.bat` resume el ledger de aprendizajes parciales seguros.

El flujo manual por pasos fue retirado: prepare/build/train/evaluate siguen existiendo como
primitivas internas, pero el entrypoint normal de aprendizaje es Night Study.

## Estado vs configuración

`config/autonomy.json` guarda defaults autonomos generales, como la receta fallback y la
fraccion de CPU permitida. No guarda target ni suite.

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

## Logs y reportes

Night Study mantiene limpia la salida runtime automaticamente:

- `logs/night-study-*.log`: conserva la sesion actual y las dos anteriores.
- `reports/brain-*-training.json`: conserva los ultimos 6.
- `reports/brain-*-evaluation.json`: conserva los ultimos 6.
- `reports/study-profile-*.json`: conserva los ultimos 6.
- `reports/lifelong/*.json`: conserva los ultimos 6.

Los `latest-*.json` se mantienen como punteros al estado mas reciente.

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
