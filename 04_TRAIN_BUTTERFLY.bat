@echo off
setlocal
cd /d "%~dp0"
if not exist ".butterfly\tokenizer-v2.json" (echo Falta el tokenizer. Ejecuta el paso 03.&pause&exit /b 1)
echo ========================================
echo  ButterflyAI v0.0003 - entrenamiento
echo ========================================
echo Perfil pensado para Ryzen 5 3600 + 16 GB RAM.
echo La RX 580 no se fuerza aqui: priorizamos un entrenamiento CPU estable.
echo No cierres esta ventana hasta que aparezca Saved.
echo.
".venv\Scripts\python.exe" -m butterfly train --version 0.0003 --preset ryzen3600 --steps 1400
if errorlevel 1 goto :fail
pause&exit /b 0
:fail
echo ERROR entrenando Butterfly.&pause&exit /b 1
