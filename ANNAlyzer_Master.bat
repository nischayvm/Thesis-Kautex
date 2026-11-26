@echo off
setlocal
pushd "%~dp0" || exit /b 1
python -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 goto python
py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 goto py
echo Python 3.10 or newer was not found. Add the installed Python to PATH or enable the py launcher.
set "STATUS=1"
goto finish
:python
python -B -u scripts\annalyzer_master.py %*
set "STATUS=%ERRORLEVEL%"
goto finish
:py
py -3 -B -u scripts\annalyzer_master.py %*
set "STATUS=%ERRORLEVEL%"
:finish
popd
if "%~1"=="" if not "%STATUS%"=="0" pause
exit /b %STATUS%
