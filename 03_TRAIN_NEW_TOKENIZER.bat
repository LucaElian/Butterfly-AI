@echo off
setlocal
cd /d "%~dp0"
if not exist "data\consolidated.txt" (echo Falta data\consolidated.txt. Ejecuta el paso 02.&pause&exit /b 1)
echo ========================================
echo  ButterflyTokenizer v2
echo ========================================
echo Aprende palabras y subpalabras frecuentes, manteniendo fallback UTF-8.
echo.
".venv\Scripts\python.exe" -m butterfly tokenizer --vocab 4096
if errorlevel 1 goto :fail
pause&exit /b 0
:fail
echo ERROR entrenando tokenizer.&pause&exit /b 1
