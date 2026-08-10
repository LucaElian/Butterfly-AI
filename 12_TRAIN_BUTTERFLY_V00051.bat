@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.00051 - CORRECTIVE TRAINING
echo ========================================
echo Seed: cerebro aceptado v0.0004 + MISMO tokenizer.
echo v0.0005 fue rechazada: sus pesos NO se reutilizan.
echo USER = contexto; loss SOLO sobre respuesta Butterfly.
echo CPU cap: 8 hilos. Autosave weights-only aprox cada 10 minutos.
echo v0.0004 sigue activa hasta aprobar benchmark v0.00042.
echo.
".venv\Scripts\python.exe" -m butterfly train-v00051 --preset ryzen3600
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR/interrupcion entrenando v0.00051.
echo v0.0004 sigue intacta. Si existe training_state\v0.00051 NO lo borres.
echo Ejecuta este BAT de nuevo para reanudar.
pause & exit /b 1
