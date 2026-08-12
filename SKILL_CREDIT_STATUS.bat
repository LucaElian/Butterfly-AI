@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: falta .venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m butterfly.learning.skill_credit --status
echo.
pause
