@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================
echo  ButterflyAI - NIGHT STUDY
echo ==================================================
echo.
echo La sesion elegira autonomamente que capacidad estudiar.
echo La sesion usa los limites configurados en config/night_study.json.
echo En modo Lifelong puede continuar hasta STOP_NIGHT_STUDY.bat.
echo Para pedir parada segura durante la noche ejecuta:
echo   STOP_NIGHT_STUDY.bat
echo.
echo Starting...
echo.

".venv\Scripts\python.exe" -m butterfly night-study
set ERR=%ERRORLEVEL%

echo.
if not "%ERR%"=="0" (
  echo Night Study termino con error %ERR%.
) else (
  echo Night Study termino.
)
echo.
pause
exit /b %ERR%
