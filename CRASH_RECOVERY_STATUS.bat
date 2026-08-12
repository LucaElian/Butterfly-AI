@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: falta .venv
  pause
  exit /b 1
)
echo ButterflyAI Crash Recovery Diagnostic
echo -------------------------------------
echo Este comando NO entrena.
echo.
".venv\Scripts\python.exe" -m butterfly night-study --dry-run
echo.
pause
