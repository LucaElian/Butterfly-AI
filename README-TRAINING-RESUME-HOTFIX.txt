ButterflyAI v0.0004 - Training Resume Hotfix

Motivo:
Un disco externo D: desaparecio justo al terminar LANGUAGE. La version previa
mantenia el mejor estado de la etapa solamente en RAM hasta el final del curriculum.

Este hotfix cambia el entrenamiento de v0.0004:
- checkpoint weights-only atomico cada ~10 minutos;
- checkpoint al final de cada epoch;
- checkpoint obligatorio al final de cada stage;
- reanuda automaticamente desde training_state/v0.0004;
- usa shuffle determinista por epoch y omite los batches ya procesados;
- el optimizer Adam NO se guarda (sus momentos reinician al reanudar), por lo que
  el recovery ocupa cerca del tamano de los pesos y no ~3x;
- la escritura es atomica: un .tmp incompleto no reemplaza el ultimo recovery bueno;
- al crear exitosamente la candidata final, borra recovery temporal para ahorrar espacio.

IMPORTANTE:
El entrenamiento LANGUAGE que se perdio antes de instalar este hotfix no puede
recuperarse porque aquella version nunca llego a escribirlo al disco. Hay que ejecutar
03_TRAIN_BUTTERFLY_V0004.bat de nuevo una vez. A partir de ahi queda protegido.
