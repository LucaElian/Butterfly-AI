@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI - exportar cerebro estable
echo ========================================
echo Crea un ZIP en release\ con SOLO pesos de inferencia + metadata + SHA256.
echo Ese ZIP se sube como asset de GitHub Release, NO con git add.
echo.
".venv\Scripts\python.exe" -m butterfly export-release
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo ERROR exportando el cerebro. Si la activa todavia es v0.0003 .pt, primero termina/promociona v0.0004.
pause & exit /b 1
