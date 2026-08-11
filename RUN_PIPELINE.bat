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
echo  ButterflyAI - PIPELINE
echo ========================================
echo.
".venv\Scripts\python.exe" -m butterfly.pipeline --mode status
echo.
echo [1] AUTOMATICO - continua desde la primera etapa pendiente
echo [2] PAUSADO    - igual que automatico, ENTER entre etapas
echo [3] UNA ETAPA  - ejecutar una etapa manualmente
echo [4] SALIR
echo.
set /p MODE=Opcion: 

if "%MODE%"=="1" goto AUTO
if "%MODE%"=="2" goto PAUSED
if "%MODE%"=="3" goto ONE
if "%MODE%"=="4" exit /b 0
goto MENU

:AUTO
".venv\Scripts\python.exe" -m butterfly.pipeline --mode auto
goto END

:PAUSED
".venv\Scripts\python.exe" -m butterfly.pipeline --mode paused
goto END

:ONE
".venv\Scripts\python.exe" -m butterfly.pipeline --mode stage-menu
goto END

:END
set CODE=%ERRORLEVEL%
echo.
echo Ultimo log: logs\latest.log
echo Resumen: reports\latest-summary.txt
pause
exit /b %CODE%
