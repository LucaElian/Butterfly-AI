@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: falta .venv. Ejecuta SETUP_WINDOWS.bat primero.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m butterfly.pipeline --mode stage --stage train
set CODE=%ERRORLEVEL%
echo.
echo Ultimo log: logs\latest.log
echo Resumen: reports\latest-summary.txt
pause
exit /b %CODE%
