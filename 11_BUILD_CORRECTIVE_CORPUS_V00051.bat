@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.00051 - CORRECTIVE CORPUS
echo ========================================
echo Construye entrenamiento robusto y contrastivo.
echo NO descarga Internet. NO usa Qwen.
echo Las frases del benchmark v0.00042 se normalizan sin tildes/puntuacion antes del anti-leak:
echo cambiar una coma o sacar signos NO alcanza para colarlas al train.
echo.
".venv\Scripts\python.exe" -m butterfly build-alignment-v00051
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR construyendo corpus. No entrenes hasta resolverlo.
pause & exit /b 1
