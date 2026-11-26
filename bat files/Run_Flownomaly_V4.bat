@echo off
setlocal
pushd "%~dp0.." || exit /b 1
set "MODEL_DIR=%~dp0..\Anamoly_detection\flownomaly"
set "PYTHON=%MODEL_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Missing the Flownomaly Python environment. See Anamoly_detection\flownomaly\README.md. 1>&2
    popd
    exit /b 1
)
"%PYTHON%" -B -u "%MODEL_DIR%\..\lighting_launcher.py" flownomaly %*
set "STATUS=%ERRORLEVEL%"
popd
exit /b %STATUS%
