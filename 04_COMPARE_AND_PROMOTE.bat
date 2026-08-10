@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI - v0.0003 VS v0.0004
echo ========================================
echo Solo si v0.0004 gana se vuelve activa y se quema el checkpoint/tokenizer viejo.
echo La memoria, corpus y benchmarks nunca se queman.
echo.
".venv\Scripts\python.exe" -m butterfly compare-promote
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo ERROR durante comparacion. No se borra el modelo activo por un fallo del script.
pause & exit /b 1
