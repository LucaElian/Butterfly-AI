@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: falta .venv. Ejecuta SETUP_WINDOWS.bat primero.
  pause
  exit /b 1
)

:MENU
cls
echo ========================================
echo  ButterflyAI - PERMANENT PIPELINE
echo ========================================
echo.
".venv\Scripts\python.exe" -m butterfly.pipeline --mode status
echo.
echo [1] AUTOMATICO  - ejecuta 01 ^> 02 ^> 03 ^> 04 sin esperar
echo [2] PAUSADO     - pide confirmacion entre etapas
echo [3] REANUDAR    - continua desde la primera etapa no completada
echo [4] UNA ETAPA   - elegir una etapa manualmente
echo [5] SALIR
echo.
set /p MODE=Opcion: 

if "%MODE%"=="1" goto AUTO
if "%MODE%"=="2" goto PAUSED
if "%MODE%"=="3" goto RESUME
if "%MODE%"=="4" goto ONE
if "%MODE%"=="5" exit /b 0
goto MENU

:AUTO
".venv\Scripts\python.exe" -m butterfly.pipeline --mode auto
goto END

:PAUSED
".venv\Scripts\python.exe" -m butterfly.pipeline --mode paused
goto END

:RESUME
".venv\Scripts\python.exe" -m butterfly.pipeline --mode resume
goto END

:ONE
echo.
echo [1] PREPARE
echo [2] BUILD DATASET
echo [3] TRAIN
echo [4] EVALUATE AND PROMOTE
set /p STAGE=Etapa: 
if "%STAGE%"=="1" ".venv\Scripts\python.exe" -m butterfly.pipeline --mode stage --stage prepare
if "%STAGE%"=="2" ".venv\Scripts\python.exe" -m butterfly.pipeline --mode stage --stage build_dataset
if "%STAGE%"=="3" ".venv\Scripts\python.exe" -m butterfly.pipeline --mode stage --stage train
if "%STAGE%"=="4" ".venv\Scripts\python.exe" -m butterfly.pipeline --mode stage --stage evaluate_and_promote
goto END

:END
echo.
echo Salida completa: logs\latest.log
echo Resumen: reports\latest-summary.txt
pause
exit /b %ERRORLEVEL%
