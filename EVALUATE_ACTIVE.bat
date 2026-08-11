@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  ButterflyAI - STRICT ACTIVE EVALUATION
echo ========================================
echo.
echo No entrena ni modifica pesos.
echo ACTIVE y evaluator se resuelven dinamicamente.
echo.

".venv\Scripts\python.exe" -m butterfly evaluate
if errorlevel 1 goto fail
pause
exit /b 0

:fail
echo.
echo ERROR during strict evaluation. No model was changed or deleted.
pause
exit /b 1
