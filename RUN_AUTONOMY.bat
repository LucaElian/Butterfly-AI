@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================
echo  ButterflyAI - AUTONOMY
echo ==================================================
echo.
echo La sesion elegira autonomamente que capacidad estudiar.
echo La sesion usa los limites configurados en config/night_study.json.
echo Puede continuar hasta que pidas parada segura con STOP_AUTONOMY.bat.
echo Para pedir parada segura durante la noche ejecuta:
echo   STOP_AUTONOMY.bat
echo.
echo Starting...
echo.

".venv\Scripts\python.exe" -m butterfly autonomy
set ERR=%ERRORLEVEL%

echo.
if not "%ERR%"=="0" (
  echo Autonomy termino con error %ERR%.
) else (
  echo Autonomy termino.
)
echo.
pause
exit /b %ERR%
