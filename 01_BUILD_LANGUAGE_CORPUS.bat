@echo off
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI v0.0004 - corpus HOTFIX
echo ========================================
echo.
echo Esta version NO pide articulos uno por uno a MediaWiki.
echo Lee Wikipedia ES en lotes de hasta 100 articulos por request desde
echo el mirror de dataset de Wikimedia en Hugging Face y conserva la URL original.
echo.
echo Lo que ya descargaste antes de frenar se conserva y cuenta para los 20 MB.
echo Si se corta, podes ejecutar este BAT de nuevo y continua con su estado local.
echo.
".venv\Scripts\python.exe" -m butterfly build-corpus --wiki-mb 20 --conversation-mb 2
if errorlevel 1 goto fail
pause & exit /b 0
:fail
echo.
echo ERROR construyendo corpus. El progreso queda guardado; podes reintentarlo.
pause & exit /b 1
