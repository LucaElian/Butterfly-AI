@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.0005 - PREPARE
echo ========================================
echo v0.0005 continua desde los pesos ACEPTADOS de v0.0004.
echo NO reinicia el cerebro, NO cambia tokenizer y NO toca memoria/corpus.
echo El benchmark estricto v0.00041 queda congelado como examen.
echo.
".venv\Scripts\python.exe" -m butterfly prepare-v0005
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR: preparacion detenida sin modificar el cerebro activo.
pause & exit /b 1
