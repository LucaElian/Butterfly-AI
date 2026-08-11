@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  ButterflyAI - WINDOWS SETUP
echo ========================================
echo.

set "PYTHON_CMD="
where python >nul 2>nul && set "PYTHON_CMD=python"
if not defined PYTHON_CMD (where py >nul 2>nul && set "PYTHON_CMD=py")
if not defined PYTHON_CMD (
  echo ERROR: No encontre Python en PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creando .venv...
  %PYTHON_CMD% -m venv .venv || goto :fail
)

".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" -m butterfly init || goto :fail
".venv\Scripts\python.exe" -m butterfly health || goto :fail
".venv\Scripts\python.exe" -m butterfly audit-hardcodes || goto :fail

echo.
echo Setup listo. No se modificaron modelos, memoria ni corpus.
pause
exit /b 0

:fail
echo.
echo ERROR: el setup fallo. Mira el mensaje de arriba.
pause
exit /b 1
