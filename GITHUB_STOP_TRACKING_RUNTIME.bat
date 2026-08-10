@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo  ButterflyAI - limpiar tracking de Git
 echo ========================================
echo.
echo Esto NO borra archivos de tu disco. Usa git rm --cached para que Git deje de versionar
 echo cerebros, memoria local y corpus generados que ahora estan cubiertos por .gitignore.
echo El tag/commit v0.0003 sigue conservando historicamente lo que ya subiste.
echo.
where git >nul 2>nul || (echo ERROR: git no esta en PATH.& pause & exit /b 1)
if not exist ".git" (echo ERROR: esta carpeta no parece ser un repositorio Git.& pause & exit /b 1)
set /p CONFIRM=Escribi LIMPIAR_GIT para continuar: 
if /I not "%CONFIRM%"=="LIMPIAR_GIT" (echo Cancelado.& pause & exit /b 0)

git rm -r --cached --ignore-unmatch models 2>nul
git rm -r --cached --ignore-unmatch training_state 2>nul
git rm -r --cached --ignore-unmatch release 2>nul
git rm --cached --ignore-unmatch .butterfly/butterfly.db 2>nul
git rm --cached --ignore-unmatch .butterfly/butterfly.db-wal 2>nul
git rm --cached --ignore-unmatch .butterfly/butterfly.db-shm 2>nul
git rm --cached --ignore-unmatch data/corpus/language_train.txt 2>nul
git rm --cached --ignore-unmatch data/corpus/language_valid.txt 2>nul
git rm --cached --ignore-unmatch data/corpus/conversation_train.txt 2>nul
git rm --cached --ignore-unmatch data/corpus/conversation_valid.txt 2>nul
git rm --cached --ignore-unmatch data/corpus/instruction_train.txt 2>nul
git rm --cached --ignore-unmatch data/corpus/instruction_valid.txt 2>nul
git rm --cached --ignore-unmatch data/corpus/wikipedia_seen.txt 2>nul
git rm --cached --ignore-unmatch data/corpus/wikipedia_sources.jsonl 2>nul

rem Restore small source-controlled placeholders/metadata if they exist.
git add .gitignore .gitattributes models/.gitkeep 2>nul
if exist "models\history.json" git add "models\history.json"
if exist "data\corpus\manifest.json" git add -f "data\corpus\manifest.json"

echo.
echo Listo. Tus archivos LOCALES siguen ahi. Revisa 'git status' antes de commit/push.
echo Los modelos viejos siguen accesibles en commits/tags anteriores; los nuevos van a GitHub Releases.
pause
