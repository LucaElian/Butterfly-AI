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


## Linux / cloud runner

Para correr ButterflyAI fuera de tu PC, por ejemplo en una VM Ubuntu de Oracle Cloud,
no hace falta instalar Linux en Windows. La VM remota corre Linux y tu PC solo se conecta
por SSH para mirar estado, logs o reportes.

Primer setup en la VM:

```bash
bash SETUP_LINUX.sh
```

Uso manual:

```bash
./TEACHER_LESSONS.sh
./RUN_AUTONOMY.sh
./STOP_AUTONOMY.sh
./AUTONOMY_STATUS.sh
```

Modo 24/7 con systemd de usuario:

```bash
./INSTALL_AUTONOMY_SERVICE.sh
systemctl --user status butterfly-autonomy.service
journalctl --user -u butterfly-autonomy.service -f
```

El servicio ejecuta `RUN_AUTONOMY_LOOP.sh`, que usa consola compacta para journalctl,
conserva los logs completos en `logs/autonomy-*.log`, intenta material local con Ollama
cuando este instalado y duerme entre sesiones segun `BUTTERFLY_AUTONOMY_LOOP_SLEEP_SECONDS`
(default: 900 segundos). Para pedir parada segura:

```bash
./STOP_AUTONOMY.sh
```

En una VM Always Free conviene empezar chico: 2 OCPU, 12 GB RAM y 100-150 GB de disco.
## Autonomy

`RUN_AUTONOMY.bat` ejecuta el aprendizaje autonomo con los limites definidos en
`config/autonomy_learning.json`. En modo Lifelong puede continuar hasta que se pida una parada
segura con `STOP_AUTONOMY.bat`.

`AUTONOMY_STATUS.bat` muestra el estado del curriculum graph y hace un chequeo rapido del
motor de examenes dinamicos.

La recuperacion de cortes y el rescate parcial seguro se ejecutan dentro de Autonomy y quedan
registrados en `autonomy-latest`, logs y benchmarks.

El flujo manual por pasos fue retirado: prepare/build/train/evaluate siguen existiendo como
primitivas internas, pero el entrypoint normal de aprendizaje es Autonomy.

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

## Logs, reportes y limpieza runtime

Autonomy mantiene limpia la salida runtime automaticamente con la politica definida en
`config/autonomy_learning.json` -> `runtime_retention`:

- `logs/autonomy-*.log`: conserva la sesion actual y las dos anteriores por defecto.
- `reports/brain-*-training.json`: conserva los ultimos 3.
- `reports/brain-*-evaluation.json`: conserva los ultimos 3.
- `reports/study-profile-*.json`: conserva los ultimos 3.
- `reports/lifelong/*.json`: conserva los ultimos 3.
- `benchmarks/comparison-*.json`: conserva los ultimos 15 no referenciados por defecto.
- `training_state/`: se limpia solo cuando no hay experimento recuperable.
- `models/butterfly-v*.safetensors`: se borran artefactos fisicos que no sean ACTIVE, LAB o CANDIDATE registrado.
- `.pytest_cache`, `.mypy_cache`, `.ruff_cache` y `__pycache__`: se limpian fuera de `.venv` y `.git`.

Los `latest-*.json` se mantienen como punteros al estado mas reciente. La limpieza no toca
memoria, configuracion reusable, registry/history, ACTIVE, LAB ni un CANDIDATE recuperable.

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
