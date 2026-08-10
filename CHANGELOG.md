# ButterflyAI — Changelog

## v0.00053 — generalization + replay candidate

- Primer target deliberado que usa el pipeline permanente `01..04` sin BATs por versión.
- Seed: cerebro aceptado v0.0004 + mismo tokenizer v3/8192.
- Nuevo benchmark v0.00043: mantiene regresiones v0.00042 y agrega surfaces/targets/pairs held-out nuevos.
- Métricas nuevas separadas: `binding_component` y `arithmetic_component`.
- `ROBUST_COMPREHENSION` protege conversación + conceptos básicos que v0.00051 había interferido.
- `COPY_BINDING` se separa de matemática y usa valores train/valid/benchmark disjuntos.
- `BASIC_ARITHMETIC` usa pares completos held-out; + y * reservan también el espejo conmutativo.
- `EPISTEMIC_CONTRAST` entrena rechazo de cuentas falsas, datos ausentes y fuentes contradictorias.
- Las etapas posteriores contienen replay de habilidades previas; `BALANCED_REPLAY` consolida todas a LR bajo.
- Assistant-only loss conserva el boundary de inferencia `Butterfly:` sin permitir que el tokenizer cruce al primer token de respuesta.
- CPU sigue limitado a 8 hilos y recovery sigue siendo weights-only atómico.

## Infrastructure cleanup — permanent pipeline

- Se reemplaza el pipeline versionado `00...13` por cuatro etapas permanentes `01...04`.
- Se agrega `RUN_PIPELINE.bat` con automático, pausado, reanudación y etapa individual.
- Toda la salida se duplica a consola + `logs/`.
- Se agrega estado local de pipeline en `.butterfly/pipeline_state.json`.
- Se agregan summaries locales en `reports/`.
- Sleep learning deja de depender de una versión hardcodeada y pasa a configuración.
- Se eliminan trainers/builders Python específicos de v0.0005/v0.00051 después de haber
  terminado y rechazado esos experimentos.
- No se elimina memoria, corpus, tokenizer aceptado, benchmarks ni historial.

## v0.00051 — rejected

Benchmark v0.00042 candidate:
- overall: 0.6378 vs baseline 0.3714;
- conversation: 0.8622;
- epistemic dialogue: 0.8967;
- robustness: 0.6126;
- contrastive: 0.5646;
- critical pass rate: 0.5625.

Mejoró conversación informal, identidad y epistemología, pero falló copy/binding,
aritmética y produjo interferencia en comprensión. La candidata física fue eliminada;
su corpus, manifest y benchmark quedaron preservados.

## v0.0005 — rejected

Benchmark v0.00041 candidate:
- overall: 0.7361 vs baseline 0.4799;
- conversation: 0.6250;
- comprehension: 0.8667;
- critical pass rate: 0.3333.

Demostró que assistant-only loss mejoraba fuertemente la relación entrada-respuesta,
pero todavía fallaba binding, matemática e intenciones cercanas.

## v0.00042 — evaluator

Benchmark más difícil con español informal/sin tildes, familias held-out,
intenciones contrastivas, targets de copy reservados y pares aritméticos reservados.

## v0.00041 — evaluator

Primer benchmark estricto que dejó de premiar principalmente estructura superficial.
La misma v0.0004 pasó de un score antiguo inflado a 0.4799 en la suite v0.00041.

## v0.0004 — active brain

- 17,477,376 parámetros.
- tokenizer v3 de 8,192.
- pretraining de lenguaje sobre millones de tokens.
- stages de language, conversation e instruction.
- modelo estable weights-only.
