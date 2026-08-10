@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.0005 - ALIGNMENT TRAINING
echo ========================================
echo Seed: cerebro aceptado v0.0004 + MISMO tokenizer.
echo Objetivo nuevo: el prompt del usuario es contexto; el loss se calcula SOLO en la respuesta Butterfly.
echo CPU cap: 8 hilos. Autosave weights-only aprox cada 10 minutos.
echo v0.0004 sigue activa hasta que v0.0005 apruebe el examen estricto.
echo.
".venv\Scripts\python.exe" -m butterfly train-v0005 --preset ryzen3600
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR/interrupcion entrenando v0.0005.
echo NO borres training_state\v0.0005: ejecuta este BAT otra vez para reanudar.
echo v0.0004 sigue activa e intacta.
pause & exit /b 1
