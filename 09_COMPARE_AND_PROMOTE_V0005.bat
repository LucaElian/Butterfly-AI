@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.0005 - STRICT EXAM
echo ========================================
echo v0.0005 NO se promueve solo por bajar loss.
echo Debe superar el baseline v0.0004 Y pasar TODOS los hard gates del benchmark v0.00041.
echo Si pierde: se elimina solo la candidata. Si gana: recien ahi se quema el cerebro fisico viejo.
echo Memoria, corpus, conocimiento verificado, historial y benchmarks siempre se conservan.
echo.
".venv\Scripts\python.exe" -m butterfly compare-promote --candidate 0.0005
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR durante comparacion. Por seguridad no borres nada manualmente.
pause & exit /b 1
