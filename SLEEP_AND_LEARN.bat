@echo off
cd /d "%~dp0"
echo Butterfly sleep cycle.
echo Safety v0.00051: no sleep candidate can bypass benchmark v0.00042 hard gates.
echo While active brain is v0.0004, sleep learning is paused because v0.00051 is the reserved corrective generation.
".venv\Scripts\python.exe" -m butterfly sleep --steps 120
pause
