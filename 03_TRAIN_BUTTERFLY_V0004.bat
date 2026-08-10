@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.0004 - curriculum RESUME
echo ========================================
echo Etapas: lenguaje - conversacion - instrucciones.
echo La candidata NO reemplaza v0.0003 hasta aprobar el examen.
echo.
echo RECUPERACION:
echo - autosave weights-only aprox. cada 10 minutos
echo - autosave al terminar cada epoch
echo - autosave obligatorio al terminar cada stage
echo - si D: desaparece o se cierra la terminal, ejecuta este BAT de nuevo
echo - no guarda Adam: ocupa mucho menos; al reanudar solo reinicia su momentum
echo.
".venv\Scripts\python.exe" -m butterfly train-v0004 --preset ryzen3600
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR/interrupcion entrenando candidata.
echo La v0.0003 sigue intacta. Si existe training_state\v0.0004,
echo NO lo borres: vuelve a ejecutar este BAT cuando el disco este disponible.
pause & exit /b 1
