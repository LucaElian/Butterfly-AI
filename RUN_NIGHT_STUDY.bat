@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================
echo  ButterflyAI - NIGHT STUDY
echo ==================================================
echo.
echo La sesion elegira autonomamente que capacidad estudiar.
echo Default: maximo 2 bloques o 180 minutos.
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
