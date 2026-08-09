@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo   ButterflyAI v0.0003 - Setup permanente
echo ========================================
echo.
set "PYTHON_CMD="
where python >nul 2>nul && set "PYTHON_CMD=python"
if not defined PYTHON_CMD (where py >nul 2>nul && set "PYTHON_CMD=py")
if not defined PYTHON_CMD (
  echo ERROR: No encontre Python en PATH.
  echo Instala Python 3.11+ y marca Add Python to PATH.
  pause & exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creando .venv...
  %PYTHON_CMD% -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" -m butterfly init || goto :fail
echo.
echo Setup listo.
echo Esta carpeta ButterflyAI es la instalacion permanente desde ahora.
pause & exit /b 0
:fail
echo.
echo ERROR: El setup fallo. Mira el mensaje de arriba.
pause & exit /b 1
