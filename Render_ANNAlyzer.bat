@echo off
setlocal
title ANNAlyzer Regular Normals v2

set "ANNALYZER_BLENDER=E:\SteamLibrary\steamapps\common\Blender\blender.exe"
set "ANNALYZER_BLEND=%~dp0synthetic_dataset\battery_tray_inspection_anomalies.blend"
set "ANNALYZER_SCRIPT=%~dp0scripts\render_annalyzer_normals_v2.py"
set "ANNALYZER_OUTPUT=%~dp0data\annalyzer_regular_normals_v2"

if not exist "%ANNALYZER_BLENDER%" goto missing
if not exist "%ANNALYZER_BLEND%" goto missing
if not exist "%ANNALYZER_SCRIPT%" goto missing

powershell.exe -NoLogo -NoProfile -Command "$output = $env:ANNALYZER_OUTPUT; $parent = Split-Path -Parent $output; New-Item -ItemType Directory -Force -Path $parent -ErrorAction Stop | Out-Null; $log = Join-Path $parent ('annalyzer_regular_normals_v2_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log'); $blenderArgs = @('--background', $env:ANNALYZER_BLEND, '--python-exit-code', '1', '--python', $env:ANNALYZER_SCRIPT, '--', '--output', $output); if (Test-Path -LiteralPath (Join-Path $output 'dataset.json') -PathType Leaf) { $blenderArgs += '--resume'; Write-Host 'Resuming the 99 regular-light normal renders.' } else { Write-Host 'Starting 99 new regular-light normal renders.' }; Write-Host ('Output: ' + $output); Write-Host ('Progress log: ' + $log); & $env:ANNALYZER_BLENDER @blenderArgs 2>&1 | Tee-Object -FilePath $log; exit $LASTEXITCODE"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
    echo Batch finished.
) else (
    echo Blender exited with code %RESULT%. Check the progress log above.
)
pause
exit /b %RESULT%

:missing
echo A required Blender file was not found. Check these paths:
echo   "%ANNALYZER_BLENDER%"
echo   "%ANNALYZER_BLEND%"
echo   "%ANNALYZER_SCRIPT%"
pause
exit /b 1
