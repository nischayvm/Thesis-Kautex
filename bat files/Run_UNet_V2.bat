@echo off
setlocal
pushd "%~dp0.." || exit /b 1
set "MODEL_DIR=%~dp0..\Anamoly_detection\unet"
set "PYTHON=%MODEL_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Missing the existing unet Python environment at "%PYTHON%". 1>&2
    popd
    exit /b 1
)
if /I "%~1"=="v2" goto v2
"%PYTHON%" -B -u "%MODEL_DIR%\..\v3_workflow.py" unet %*
goto finish
:v2
shift
if "%~1"=="" (
    "%PYTHON%" -B -u "%MODEL_DIR%\V2\run_v2.py" validate --use-reviewed-split
    goto finish
)
"%PYTHON%" -B -u "%MODEL_DIR%\V2\run_v2.py" %1 %2 %3 %4 %5 %6 %7 %8 %9
:finish
set "STATUS=%ERRORLEVEL%"
popd
exit /b %STATUS%
