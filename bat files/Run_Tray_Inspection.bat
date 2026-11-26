@echo off
setlocal
set "APP=%~dp0..\Anamoly_detection\inspection_app"
if not exist "%APP%\.venv\Scripts\python.exe" (
    echo Tray inspection UI environment is missing. See Anamoly_detection\inspection_app\README.md.
    pause
    exit /b 1
)
"%APP%\.venv\Scripts\python.exe" -B "%APP%\app.py"
if errorlevel 1 pause
