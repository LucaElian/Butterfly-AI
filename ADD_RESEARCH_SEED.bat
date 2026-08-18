@echo off
setlocal
cd /d "%~dp0"
echo ButterflyAI - carga automatica de material verificado base
echo.
echo Esto agrega experiencias verificadas curadas con procedencia.
echo No descarga internet ni entrena texto crudo directamente.
echo.
.venv\Scripts\python.exe -m butterfly research-seed
if errorlevel 1 (
  echo.
  echo No se pudo cargar el research seed.
  pause
  exit /b 1
)
echo.
echo Listo. El proximo RUN_AUTONOMY puede usar este material cuando toque esos nodos.
pause