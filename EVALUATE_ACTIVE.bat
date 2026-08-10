@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.00041 - STRICT EVALUATION
echo ========================================
echo This does NOT train or modify the active brain.
echo It evaluates semantics, basic dialogue, instruction following,
echo epistemic behavior, coherence and hard promotion gates.
echo.
".venv\Scripts\python.exe" -m butterfly evaluate
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR during strict evaluation. No model was changed or deleted.
pause & exit /b 1
