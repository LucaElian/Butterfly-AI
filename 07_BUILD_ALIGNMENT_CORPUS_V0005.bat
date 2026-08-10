@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.0005 - ALIGNMENT CORPUS
echo ========================================
echo Construye dialogos limpios para entrada - comprension - respuesta.
echo No usa Qwen ni descarga Internet.
echo Los prompts EXACTOS del benchmark v0.00041 estan prohibidos en este corpus.
echo.
".venv\Scripts\python.exe" -m butterfly build-alignment-v0005
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR construyendo el corpus. v0.0004 sigue intacta.
pause & exit /b 1
