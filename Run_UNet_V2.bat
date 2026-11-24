@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "MODEL_DIR=%~dp0Anamoly_detection\unet"
set "PYTHON=%MODEL_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Missing the existing unet Python environment at "%PYTHON%". 1>&2
    popd
    exit /b 1
)
set "DEFAULT_ARGS="
if "%~1"=="" set "DEFAULT_ARGS=validate --use-reviewed-split"
"%PYTHON%" -B -u "%MODEL_DIR%\V2\run_v2.py" %DEFAULT_ARGS% %*
set "STATUS=%ERRORLEVEL%"
popd
exit /b %STATUS%
