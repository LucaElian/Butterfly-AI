@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".butterfly" mkdir ".butterfly"
type nul > ".butterfly\STOP_AUTONOMY"
echo Stop requested.
echo Butterfly terminara de forma segura apenas llegue al proximo punto seguro.
pause
