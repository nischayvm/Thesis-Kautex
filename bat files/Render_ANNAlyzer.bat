@echo off
setlocal
title ANNAlyzer Render

set "ANNALYZER_BLENDER=E:\SteamLibrary\steamapps\common\Blender\blender.exe"
set "ANNALYZER_BLEND=%~dp0..\synthetic_dataset\battery_tray_inspection_anomalies.blend"
if /I "%~1"=="v1" goto v1
if /I "%~1"=="v2" goto v2
if /I "%~1"=="--status" goto status
if not "%~1"=="" goto usage
echo.
echo 1. Regular-light normal supplement ^(V2^)
echo 2. Three-light original batch ^(V1: Red, Green, Regular^)
echo 3. Check saved render status
echo Q. Cancel
choice /c 123Q /n /m "Select a render job: "
if errorlevel 255 exit /b 1
if errorlevel 4 exit /b 0
if errorlevel 3 goto status
if errorlevel 2 goto v1
if errorlevel 1 goto v2
exit /b 1

:v1
set "ANNALYZER_JOB=Original V1 three-light batch"
set "ANNALYZER_SCRIPT=%~dp0..\scripts\render_annalyzer_batch.py"
set "ANNALYZER_OUTPUT=%~dp0..\data\annalyzer_batch_2048"
goto launch

:v2
set "ANNALYZER_JOB=V2 regular-light normal supplement"
set "ANNALYZER_SCRIPT=%~dp0..\scripts\render_annalyzer_normals_v2.py"
set "ANNALYZER_OUTPUT=%~dp0..\data\annalyzer_regular_normals_v2"

:launch
if not exist "%ANNALYZER_BLENDER%" goto missing
if not exist "%ANNALYZER_BLEND%" goto missing
if not exist "%ANNALYZER_SCRIPT%" goto missing

powershell.exe -NoLogo -NoProfile -Command "$output = $env:ANNALYZER_OUTPUT; $state = Join-Path $output 'dataset.json'; if ((Test-Path -LiteralPath $state -PathType Leaf) -and ((Get-Content -LiteralPath $state -Raw | ConvertFrom-Json).complete -eq $true)) { Write-Host ('Already complete: ' + $output); exit 0 }; $parent = Split-Path -Parent $output; New-Item -ItemType Directory -Force -Path $parent -ErrorAction Stop | Out-Null; $log = Join-Path $parent ((Split-Path -Leaf $output) + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log'); $blenderArgs = @('--background', $env:ANNALYZER_BLEND, '--python-exit-code', '1', '--python', $env:ANNALYZER_SCRIPT, '--', '--output', $output); if (Test-Path -LiteralPath (Join-Path $output 'dataset.json') -PathType Leaf) { $blenderArgs += '--resume'; Write-Host ('Resuming ' + $env:ANNALYZER_JOB) } else { Write-Host ('Starting ' + $env:ANNALYZER_JOB) }; Write-Host ('Output: ' + $output); Write-Host ('Progress log: ' + $log); & $env:ANNALYZER_BLENDER @blenderArgs 2>&1 | Tee-Object -FilePath $log; exit $LASTEXITCODE"
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

:status
set "ANNALYZER_DATA=%~dp0..\data"
powershell.exe -NoLogo -NoProfile -Command "foreach ($name in @('annalyzer_batch_2048','annalyzer_regular_normals_v2')) { $file = Join-Path (Join-Path $env:ANNALYZER_DATA $name) 'dataset.json'; $status = 'not started'; if (Test-Path -LiteralPath $file -PathType Leaf) { $record = Get-Content -LiteralPath $file -Raw | ConvertFrom-Json; if ($record.complete) { $status = 'complete' } else { $status = 'incomplete' } }; Write-Host ($name + ': ' + $status) }"
if "%~1"=="" pause
exit /b 0

:usage
echo Usage: Render_ANNAlyzer.bat [v1 ^| v2 ^| --status]
exit /b 2
