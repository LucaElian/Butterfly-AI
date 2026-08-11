@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".butterfly" mkdir ".butterfly"
type nul > ".butterfly\STOP_NIGHT_STUDY"
echo Stop requested.
echo Butterfly terminara de forma segura antes del proximo bloque.
pause
