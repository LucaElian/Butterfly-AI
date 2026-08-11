@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================
echo  ButterflyAI - NIGHT STUDY V2 DIAGNOSTIC
echo ==================================================
echo.
echo Solo recopila el codigo/config local relevante.
echo NO modifica el proyecto, NO crea experimento y NO entrena.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: falta .venv
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "_collect_night_study_v2_diagnostic.py"
if errorlevel 1 goto fail

echo.
echo ==================================================
echo  DIAGNOSTIC OK
echo ==================================================
echo.
echo Subime:
echo   reports\night-study-v2-diagnostic.zip
echo.
goto cleanup

:fail
echo.
echo ERROR recopilando diagnostico.
pause
exit /b 1

:cleanup
del /q "_collect_night_study_v2_diagnostic.py" 2>nul
echo.
pause
start "" /b cmd /c "timeout /t 1 /nobreak >nul & del /q ""%~f0"""
exit /b 0
