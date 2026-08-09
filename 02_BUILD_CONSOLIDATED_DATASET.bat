@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Primero ejecuta SETUP_WINDOWS.bat&pause&exit /b 1)
echo ========================================
echo  Butterfly - nuevo material de estudio
echo ========================================
echo Incluye herencia + identidad + mucha mas conversacion y lenguaje.
echo El profesor Qwen se usa solo para generar/corregir ejemplos.
echo.
".venv\Scripts\python.exe" -m butterfly build-data --examples 600 --batch 4
if errorlevel 1 goto :fail
echo.
echo Dataset consolidado listo.
pause&exit /b 0
:fail
echo ERROR creando dataset.&pause&exit /b 1
