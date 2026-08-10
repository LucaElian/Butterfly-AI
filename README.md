# ButterflyAI v0.0005 — alignment sobre el cerebro v0.0004

ButterflyAI usa una **instalación permanente**. v0.0005 no crea otra carpeta y no reinicia el modelo desde cero.

## Estado heredado

Antes de esta actualización:

- software/sistema: v0.00041;
- cerebro activo: v0.0004;
- arquitectura: 17,477,376 parámetros;
- tokenizer v3: 8,192 tokens;
- benchmark estricto v0.00041 de v0.0004: `0.4799` overall;
- semantic: `0.2778`;
- conversation: `0.0936`;
- critical pass rate: `0.0000`.

v0.0005 conserva el cerebro v0.0004 como baseline físico hasta que la candidata apruebe el examen.

## Qué cambia en v0.0005

El problema de v0.0004 no era principalmente falta de vocabulario: podía producir texto con apariencia de español, pero respondía mal a la intención del usuario.

v0.0005 mantiene **el mismo Transformer y el mismo tokenizer** y hace continued learning sobre los pesos aceptados de v0.0004.

El cambio principal es el objetivo de entrenamiento:

```text
User: mensaje     -> CONTEXTO, no paga loss
Butterfly: salida -> SOLO estos tokens pagan loss
```

Así el modelo deja de gastar parte del entrenamiento en aprender a predecir el texto del usuario y se concentra en **qué debe responder** dado ese contexto.

También corrige una diferencia sutil de v0.0004: el límite `Butterfly:` se tokeniza durante v0.0005 exactamente como se usa en chat/evaluación, y el espacio inicial de la respuesta se aprende como salida. Esto reduce el desajuste entre formato de entrenamiento e inferencia.

## Curriculum v0.0005

1. `BASIC_DIALOGUE`: saludos, agradecimientos, identidad, aclaraciones, definiciones y respuestas directas.
2. `INSTRUCTION_FOLLOWING`: una palabra/número exacto, aritmética, formato y restricciones.
3. `EPISTEMIC_DIALOGUE`: rechazar cuentas falsas, no inventar datos ausentes y comparar evidencia.
4. `MIXED_CONSOLIDATION`: mezcla balanceada a learning rate bajo para consolidar sin olvidar las etapas anteriores.

En las primeras tres etapas se congelan los 3 bloques inferiores del Transformer. En la consolidación se habilita todo el modelo con learning rate más bajo.

## Benchmark held-out

Los prompts exactos del benchmark estricto v0.00041 **no pueden aparecer** en el corpus v0.0005. El builder lo comprueba automáticamente y también verifica que train y validation no compartan prompts.

Esto evita entrenar a Butterfly memorizando directamente las preguntas del examen.

## Orden de ejecución

```text
06_PREPARE_V0005.bat
07_BUILD_ALIGNMENT_CORPUS_V0005.bat
08_TRAIN_BUTTERFLY_V0005.bat
09_COMPARE_AND_PROMOTE_V0005.bat
```

No vuelvas a ejecutar `00/01/02/03` para crear v0.0005. Wikipedia y el tokenizer v3 ya existen y se heredan.

### 06_PREPARE_V0005

Comprueba que:

- v0.0004 sigue activa;
- modelo y tokenizer coinciden;
- existe/puede recrearse el baseline estricto v0.00041;
- los pesos activos permanecen read-only.

### 07_BUILD_ALIGNMENT_CORPUS_V0005

Genera localmente datasets JSONL limpios y balanceados. No usa Qwen ni descarga Internet.

### 08_TRAIN_BUTTERFLY_V0005

Carga los pesos **aceptados de v0.0004** y continúa el aprendizaje. Usa:

- assistant-only supervised loss;
- CPU cap de 8 hilos en Windows;
- autosaves weights-only atómicos aproximadamente cada 10 minutos;
- resume por stage/epoch/batch;
- validación y best checkpoint por etapa.

v0.0004 sigue siendo el cerebro activo durante todo el entrenamiento.

### 09_COMPARE_AND_PROMOTE_V0005

v0.0005 solo se vuelve activa si:

- mejora por al menos `+0.03` el overall de v0.0004;
- pasa **todos** los critical hard gates;
- alcanza los umbrales semánticos, conversacionales, de comprensión, instrucciones y epistemología;
- no produce una regresión importante en capacidades principales;
- su metadata confirma que realmente desciende de la v0.0004 activa y usa su mismo tokenizer.

Si pierde, se borra únicamente la candidata. Si gana, recién después de la promoción se elimina el cerebro físico anterior. Memoria, corpus, conocimientos verificados, fuentes, benchmarks e historial permanecen.

## Sleep cycle

Mientras v0.0004 sea activa, `SLEEP_AND_LEARN.bat` queda pausado para que un sleep candidate no ocupe accidentalmente la versión `0.0005`. Una vez superada esa migración, el sleep cycle también usa los hard gates estrictos antes de poder promover un cerebro.

## Almacenamiento

Los modelos estables siguen siendo `.safetensors` weights-only. El estado de Adam no forma parte del cerebro aceptado. Los checkpoints de recuperación son temporales y se eliminan cuando la candidata final se guarda correctamente.

## v0.00051 — robust corrective alignment

The active seed remains v0.0004 after the rejected v0.0005 experiment. v0.00051 adds benchmark suite v0.00042, family-held-out validation, casual/punctuationless Spanish, contrastive intent training, variable binding/copy training and held-out arithmetic combinations. Promotion still requires score improvement **and** every strict hard gate.
