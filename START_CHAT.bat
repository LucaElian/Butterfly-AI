@echo off
setlocal
cd /d "%~dp0"
if not exist "models\registry.json" (echo Butterfly aun no tiene un modelo activo.&pause&exit /b 1)
".venv\Scripts\python.exe" -m butterfly chat
pause
