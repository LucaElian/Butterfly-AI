@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m butterfly verify "2 + 2 = 5"
pause
