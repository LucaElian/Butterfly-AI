@echo off
setlocal
cd /d "%~dp0"
echo Butterfly creara una candidata usando experiencias verificadas.
echo Solo si supera la evaluacion se promociona; entonces se borra el checkpoint anterior.
".venv\Scripts\python.exe" -m butterfly sleep --steps 120
pause
