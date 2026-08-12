@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: falta .venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m butterfly.learning.curriculum_graph --status
echo.
echo Dynamic Exam quick check:
".venv\Scripts\python.exe" -m butterfly.learning.dynamic_exam --self-test
echo.
pause
