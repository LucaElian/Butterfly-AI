@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo  ButterflyAI - LOCAL TEACHER LESSONS
echo ==================================================
echo.
echo Usa Ollama local. No usa OpenAI, Gemini ni APIs pagas.
echo Genera material chico y lo guarda como experiencias verificadas.
echo.

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" -m butterfly teacher-lessons
if errorlevel 1 (
  echo.
  echo Teacher termino con error.
  echo Si Ollama no esta listo, instala/abre Ollama y ejecuta:
  echo   ollama pull qwen2.5:3b
  echo Despues volve a ejecutar TEACHER_LESSONS.bat
)

echo.
pause
endlocal
