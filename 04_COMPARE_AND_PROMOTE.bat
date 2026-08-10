@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI - ACTIVE VS CANDIDATE
echo  Strict promotion suite v0.00041
echo ========================================
echo A candidate can only be promoted if it:
echo   1. beats the active brain by the required score margin,
echo   2. passes semantic/basic-dialogue hard gates,
echo   3. has no major capability regression.
echo.
echo SAFETY: the active brain can never be deleted as a rejected candidate.
echo If no candidate exists, this BAT only reports that fact.
echo.
".venv\Scripts\python.exe" -m butterfly compare-promote
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR during comparison. The active brain is protected and is not deleted.
pause & exit /b 1
