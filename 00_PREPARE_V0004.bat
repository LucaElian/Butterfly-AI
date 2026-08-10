@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.0004 - baseline seguro
 echo ========================================
echo Evalua v0.0003 con el examen nuevo y preserva su tokenizer.
echo Si el viejo .pt contiene Adam, crea una copia safetensors weights-only,
echo verifica peso por peso y SOLO entonces elimina el .pt local pesado.
echo No cambia la version ni la inteligencia de v0.0003.
echo.
".venv\Scripts\python.exe" -m butterfly prepare-v0004
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo ERROR. No continues con el upgrade hasta resolverlo.
pause & exit /b 1
