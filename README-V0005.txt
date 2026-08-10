ButterflyAI v0.0005 — ALIGNMENT GENERATION
==========================================

Objetivo
--------
v0.0004 ya aprendio bastante forma de lenguaje, pero el benchmark estricto v0.00041
mostro el problema real: semantic=0.2778, conversation=0.0936 y critical=0.0000.

v0.0005 NO agranda el modelo y NO empieza de cero. Continua desde los pesos aceptados
de v0.0004 (17,477,376 parametros) y usa exactamente el mismo tokenizer v3 de 8,192.

Cambio principal de entrenamiento
---------------------------------
v0.0004 entrenaba un stream causal completo:
  User: pregunta -> Butterfly: respuesta
Eso hace que parte del loss se gaste tambien en predecir el texto del usuario.

v0.0005 usa assistant-only supervised loss:
  - el mensaje User se entrega como CONTEXTO;
  - sus tokens quedan enmascarados con target=-100;
  - el loss se calcula SOLO sobre los tokens que Butterfly debe responder;
  - tambien aprende el marcador de fin para cortar la respuesta.

Curriculum
----------
1. BASIC_DIALOGUE
   saludos, agradecimientos, identidad, estado, aclaraciones y definiciones directas.
2. INSTRUCTION_FOLLOWING
   una palabra exacta, numeros exactos, aritmetica, formato y restricciones.
3. EPISTEMIC_DIALOGUE
   rechazar cuentas falsas, no inventar datos ausentes y contrastar fuentes.
4. MIXED_CONSOLIDATION
   mezcla balanceada de las habilidades anteriores con learning rate bajo.

En los primeros tres stages se congelan los 3 bloques inferiores del Transformer para
proteger parte del lenguaje ya aprendido. La consolidacion final vuelve a habilitar todo
el modelo con learning rate bajo.

Anti-trampa del benchmark
-------------------------
El generador de corpus importa los prompts del benchmark estricto v0.00041 y PROHIBE
que cualquiera de esos prompts exactos aparezca en train o validation. Tambien comprueba
que no exista overlap de prompts entre train y validation.

Orden
-----
06_PREPARE_V0005.bat
07_BUILD_ALIGNMENT_CORPUS_V0005.bat
08_TRAIN_BUTTERFLY_V0005.bat
09_COMPARE_AND_PROMOTE_V0005.bat

NO volver a ejecutar 00/01/02/03 para v0.0005.
No se reconstruye Wikipedia ni el tokenizer.

Promocion
---------
El benchmark sigue siendo suite v0.00041. v0.0005 solo puede ser promovida si:
  - mejora al menos +0.03 sobre la v0.0004 activa;
  - pasa todos los critical hard gates;
  - alcanza los umbrales semanticos/conversacionales/instruccionales;
  - no tiene una regresion importante respecto de v0.0004.

La candidata rechazada se elimina. La aceptada reemplaza fisicamente al viejo cerebro
solo DESPUES de aprobar. Corpus, memoria, benchmarks e historial nunca se queman.
