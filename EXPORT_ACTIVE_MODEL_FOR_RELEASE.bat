@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  ButterflyAI - EXPORT ACTIVE MODEL
echo ========================================
echo.
echo Exporta dinamicamente el cerebro ACTIVE registrado.
echo Incluye solo pesos de inferencia, metadata, tokenizer y hashes.
echo.

".venv\Scripts\python.exe" -m butterfly export-release
if errorlevel 1 goto fail
pause
exit /b 0

:fail
echo.
echo ERROR exportando el cerebro ACTIVE.
echo Ejecuta STATUS.bat y STORAGE_STATUS.bat para revisar el estado actual.
pause
exit /b 1
