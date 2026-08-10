@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.00041 - RE-EVALUATE ACTIVE
echo ========================================
echo Active brain weights are READ ONLY in this step.
echo Expected active brain after v0.0004 promotion: v0.0004.
echo.
".venv\Scripts\python.exe" -m butterfly evaluate
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR during evaluation. Active brain remains untouched.
pause & exit /b 1
