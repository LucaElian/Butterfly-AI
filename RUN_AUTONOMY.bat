@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
cls

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: falta .venv. Ejecuta SETUP_WINDOWS.bat primero.
  pause
  exit /b 1
)

if not defined BUTTERFLY_AUTONOMY_CONSOLE set "BUTTERFLY_AUTONOMY_CONSOLE=live"
if not defined BUTTERFLY_AUTONOMY_PAUSE set "BUTTERFLY_AUTONOMY_PAUSE=never"

echo ==================================================
echo  ButterflyAI - AUTONOMY
echo ==================================================
echo.
echo Consola: %BUTTERFLY_AUTONOMY_CONSOLE% ^(el log guarda todo completo^).
echo Objetivo: estudiar capacidades pendientes sin supervision constante.
echo Seed/LAB/ACTIVE, suite y target se resuelven dinamicamente.
echo Limites: config/autonomy_learning.json ^(0 = ilimitado^).
echo STOP seguro: STOP_AUTONOMY.bat.
echo.
echo Durante la sesion vas a ver progreso vivo y eventos importantes:
echo   BLOCK        intento autonomo actual.
echo   EPOCH        avance de entrenamiento y answer-loss en vivo.
echo   LAB_ACCEPTED mejora conservada como nuevo LAB.
echo   REJECTED     pesos descartados; evidencia queda en benchmark.
echo   SKILL CREDIT rescate parcial seguro sin aceptar pesos rotos.
echo   TEACHER      material local verificado generado con Ollama si falta corpus.
echo.
echo Starting...
echo.

"%PY%" -m butterfly autonomy
set ERR=%ERRORLEVEL%

echo.
if not "%ERR%"=="0" (
  echo Autonomy termino con error %ERR%.
  if /I not "%BUTTERFLY_AUTONOMY_PAUSE%"=="never" pause
) else (
  echo Autonomy termino. Revisa reports\autonomy-latest.json o logs\autonomy-*.log para el detalle completo.
  if /I "%BUTTERFLY_AUTONOMY_PAUSE%"=="always" pause
)
exit /b %ERR%
