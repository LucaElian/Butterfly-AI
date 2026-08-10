@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo   ButterflyAI v0.0004 - Update setup
echo ========================================
echo.
set "PYTHON_CMD="
where python >nul 2>nul && set "PYTHON_CMD=python"
if not defined PYTHON_CMD (where py >nul 2>nul && set "PYTHON_CMD=py")
if not defined PYTHON_CMD (
  echo ERROR: No encontre Python en PATH.
  pause & exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creando .venv...
  %PYTHON_CMD% -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" -m butterfly init || goto :fail
rem Remove obsolete v0.0003 launchers so the permanent folder only shows the current pipeline.
del /q "01_MIGRATE_PREVIOUS_AND_CLEAN.bat" 2>nul
del /q "02_BUILD_CONSOLIDATED_DATASET.bat" 2>nul
del /q "03_TRAIN_NEW_TOKENIZER.bat" 2>nul
del /q "04_TRAIN_BUTTERFLY.bat" 2>nul
del /q "05_EVALUATE_BUTTERFLY.bat" 2>nul
rmdir /s /q "butterfly\distillation" 2>nul
del /q "butterfly\migration.py" 2>nul
echo.
echo Setup v0.0004 listo. Tus modelos, memoria y corpus anteriores no se borraron.
pause & exit /b 0
:fail
echo ERROR: El setup fallo. Mira el mensaje de arriba.
pause & exit /b 1
