@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ButterflyAI sleep cycle.
echo El estado, seed, candidato y evaluator se resuelven dinamicamente.
echo.
".venv\Scripts\python.exe" -m butterfly sleep --steps 120
pause
