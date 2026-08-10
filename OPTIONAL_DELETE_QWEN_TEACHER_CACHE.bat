@echo off
setlocal
set "TARGET=%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B"
echo Butterfly v0.0004 ya no necesita Qwen para construir el corpus principal.
echo Este script SOLO borra el cache local de Qwen3-0.6B para recuperar espacio.
echo Si usas ese modelo en otro proyecto, NO lo borres.
echo.
echo Ruta: %TARGET%
if not exist "%TARGET%" (
 echo No encontre ese cache. Nada que borrar.
 pause & exit /b 0
)
set /p CONFIRM=Escribi BORRAR_QWEN para eliminarlo: 
if /I not "%CONFIRM%"=="BORRAR_QWEN" (
 echo Cancelado.
 pause & exit /b 0
)
rmdir /s /q "%TARGET%"
echo Cache Qwen eliminado.
pause
