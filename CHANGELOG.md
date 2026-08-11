# ButterflyAI — Changelog

## General runtime-state refactor

- Brain and evaluator versions are no longer hardcoded in operational code.
- Runtime registry now has explicit ACTIVE / LAB / CANDIDATE slots.
- Pipeline AUTOMATICO and PAUSADO resume from the first incomplete stage by default.
- REANUDAR was removed as a separate mode.
- Logs are stage-specific and include the dynamic target in their filename.
- Pipeline stage completion is tied to input signatures; changed inputs invalidate downstream work.
- Experiment target versions are allocated from registry/history.
- Training recipes are reusable and do not contain target/active/suite versions.
- Evaluator suite identity is a fingerprint of cases, gates and semantic rules.
- Historical curriculum trainer was removed; general training runtime helpers remain.
- Corpus building is modularized into reusable skills.
- Promotion supports CANDIDATE -> LAB and CANDIDATE/LAB -> ACTIVE.
- A hardcode audit is part of the permanent test/setup flow.

## v0.00044 — evaluator fairness + intent routing

- No crea ni entrena un cerebro nuevo.
- Suite estricta actual: `v0.00044`.
- Corrige falsos negativos semánticos observados en v0.00053:
  - una definición correcta de API ya no depende del stem literal `comunic`;
  - una definición correcta de parámetro puede aprovechar el contexto del prompt y no
    tiene que repetir mecánicamente la palabra `parámetro`.
- File/folder/API/parameter/token usan validadores conceptuales más claros.
- Se agrega `intent_routing_component` como hard gate separado (`>= 0.75`).
- Exact-output, arithmetic, epistemic, robustness y contrastive de v0.00043 se conservan.
- El pipeline deliberado queda sin target hasta instalar intencionalmente la próxima candidata.
- Los tests operativos del evaluador pasan a nombres permanentes, no versionados.

## v0.00053 — rejected (resultado real)

- Baseline v0.0004 / suite v0.00043: `0.3114`.
- Candidate overall: `0.5534` (`+0.2421`), pero hard gates `FAIL`.
- `comprehension_component`: `0.9364`.
- `epistemic_dialogue_component`: `0.7810`.
- `binding_component`: `0.2857`.
- `arithmetic_component`: `0.0000`.
- `conversation_component`: `0.4738`.
- La candidata física fue eliminada; corpus, benchmark, tokenizer, memoria e historial se conservaron.
- Lección principal: comprensión conceptual sí mejoró, pero intent routing, exact copy y
  aritmética no generalizaron lo suficiente.

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
