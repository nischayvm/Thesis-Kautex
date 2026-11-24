@echo off
setlocal
set "RUNNER=%~dp0scripts\publish_v2_results.py"
set "PYTHON=%~dp0Anamoly_detection\patchcore\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" -B -u "%RUNNER%" --update-journal %*
exit /b %ERRORLEVEL%
