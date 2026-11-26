@echo off
setlocal
set "RUNNER=%~dp0..\scripts\publish_v2_results.py"
set "PYTHON=%~dp0..\Anamoly_detection\patchcore\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
set "MENU_ARGS="
if "%~1"=="" set "MENU_ARGS=--interactive"
"%PYTHON%" -B -u "%RUNNER%" --update-journal %MENU_ARGS% %*
exit /b %ERRORLEVEL%
