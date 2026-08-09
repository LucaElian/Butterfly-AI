@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Primero ejecuta SETUP_WINDOWS.bat&pause&exit /b 1)
echo ========================================
echo  Heredar Butterfly anterior + quemarla
echo ========================================
echo.
echo Pega o arrastra aqui la carpeta de ButterflyAI-v0.0002 y presiona Enter.
set /p "OLD=Carpeta anterior: "
if "%OLD%"=="" (echo Cancelado.&pause&exit /b 1)
".venv\Scripts\python.exe" -m butterfly migrate --previous "%OLD%" --burn
if errorlevel 1 goto :fail
echo.
echo Herencia terminada.
pause&exit /b 0
:fail
echo ERROR durante la migracion. NO se borro nada si la verificacion no termino.
pause&exit /b 1
