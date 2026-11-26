@echo off
setlocal
title ANNAlyzer Demo Image Renderer

set "DEMO_BLENDER=E:\SteamLibrary\steamapps\common\Blender\blender.exe"
set "DEMO_BLEND=%~dp0..\synthetic_dataset\battery_tray_inspection_anomalies.blend"
set "DEMO_SCRIPT=%~dp0..\scripts\render_annalyzer_demo.py"

if not exist "%DEMO_BLENDER%" goto missing
if not exist "%DEMO_BLEND%" goto missing
if not exist "%DEMO_SCRIPT%" goto missing

:menu
echo.
echo ANNAlyzer Demo Image Renderer
echo.
echo   1. Regular lighting
echo   2. Red rig lighting
echo   3. Green rig lighting
echo   Q. Quit
choice /c 123Q /n /m "Choose lighting: "
set "SELECTION=%ERRORLEVEL%"
set "DEMO_LIGHT="
if "%SELECTION%"=="1" set "DEMO_LIGHT=regular"
if "%SELECTION%"=="2" set "DEMO_LIGHT=red"
if "%SELECTION%"=="3" set "DEMO_LIGHT=green"
if not defined DEMO_LIGHT exit /b 0

echo.
echo   1. Clean tray and all-black binary mask
echo   2. Blistered tray and binary mask
echo   Q. Quit
choice /c 12Q /n /m "Choose image type: "
set "SELECTION=%ERRORLEVEL%"
set "DEMO_KIND="
if "%SELECTION%"=="1" set "DEMO_KIND=clean"
if "%SELECTION%"=="2" set "DEMO_KIND=blistered"
if not defined DEMO_KIND exit /b 0

set "ANNALYZER_DEMO_SEED="
if /i not "%DEMO_KIND%"=="clean" set /p "ANNALYZER_DEMO_SEED=Enter a blister seed, or press Enter for a random seed: "
echo.
"%DEMO_BLENDER%" --background "%DEMO_BLEND%" --python-exit-code 1 --python "%DEMO_SCRIPT%" -- --lighting "%DEMO_LIGHT%" --kind "%DEMO_KIND%"
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo Render failed with exit code %RESULT%. See the Blender message above.
choice /c YN /n /m "Generate another demo sample [Y/N]? "
if "%ERRORLEVEL%"=="1" goto menu
exit /b %RESULT%

:missing
echo A required file is missing. Check the following paths:
echo   "%DEMO_BLENDER%"
echo   "%DEMO_BLEND%"
echo   "%DEMO_SCRIPT%"
pause
exit /b 1
