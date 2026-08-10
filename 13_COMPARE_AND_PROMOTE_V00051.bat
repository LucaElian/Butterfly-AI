@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.00051 - HARD EXAM v0.00042
echo ========================================
echo No alcanza con superar el score del baseline.
echo Debe pasar TODOS los hard gates, incluyendo variantes informales y contrastivas.
echo Si pierde: se elimina SOLO la candidata v0.00051.
echo Si gana: recien ahi v0.00051 se vuelve activa y se compacta el cerebro fisico viejo.
echo Memoria, corpus, conocimiento verificado, historial y benchmarks siempre se conservan.
echo.
".venv\Scripts\python.exe" -m butterfly compare-promote --candidate 0.00051
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR durante comparacion. Por seguridad no borres nada manualmente.
pause & exit /b 1
