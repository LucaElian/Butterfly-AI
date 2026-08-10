@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.00051 - PREFLIGHT + v0.00042 BASELINE
echo ========================================
echo No entrena ni modifica pesos.
echo Reevalua el cerebro ACTIVO v0.0004 con un examen mas dificil:
echo - español informal / sin tildes / poca puntuacion
echo - intenciones parecidas (nombre vs estado)
echo - copy/binding con valores no vistos
echo - matematica con pares reservados
echo - epistemologia con variantes nuevas
echo.
".venv\Scripts\python.exe" -m butterfly prepare-v00051
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR en preflight. No se modificaron los pesos.
pause & exit /b 1
