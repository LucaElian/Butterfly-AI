@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.0004 - SUBWORD 8192
echo ========================================
echo Entrenando el nuevo vocabulario sobre el corpus acumulado...
".venv\Scripts\python.exe" -m butterfly train-tokenizer --vocab 8192
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo ERROR entrenando tokenizer.
pause & exit /b 1
