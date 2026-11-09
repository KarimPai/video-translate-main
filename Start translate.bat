@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title Japanese Video Translator
cd /d "%~dp0"

if errorlevel 1 (
    echo ERROR: Could not enter the project directory.
    pause
    exit /b 1
)

echo.
echo ========================================
echo       Japanese Video Translator
echo ========================================
echo.

set "VIDEO_URL="
set /p "VIDEO_URL=Paste YouTube live or VOD URL: "

if not defined VIDEO_URL (
    echo No URL entered.
    pause
    exit /b 1
)

if not exist "logs" mkdir "logs"

echo.
echo Starting translation with MPV (paused)...
echo.

rem Load this checkout even when the project's generated console launcher is stale.
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)
rem GTCRN and CPU speaker segmentation share sherpa-onnx. Streaming GTCRN
rem requires newer sherpa-onnx (OnlineSpeechDenoiser); upgrade old installs once.
set "NEED_SYNC=0"
if not exist "uv.lock" set "NEED_SYNC=1"
uv run --no-sync python -c "import sherpa_onnx; assert hasattr(sherpa_onnx, 'OnlineSpeechDenoiser') and hasattr(sherpa_onnx, 'OnlineSpeechDenoiserConfig')" >nul 2>nul
if errorlevel 1 set "NEED_SYNC=1"
if "%NEED_SYNC%"=="1" (
    echo Installing GTCRN and CPU speaker runtime...
    uv sync --extra diarization
    if errorlevel 1 (
        echo ERROR: Could not install GTCRN and speaker dependencies.
        pause
        exit /b 1
    )
)

uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --delay 10 --no-web --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: Translator exited with code %EXIT_CODE%.
    echo Last startup messages:
    powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\run.log' -Encoding UTF8 -Tail 6 | ForEach-Object { if ($_.Length -gt 350) { $_.Substring(0,350) } else { $_ } }"
) else (
    echo Translator stopped.
)
echo Full diagnostic log: "%CD%\logs\run.log"
pause
exit /b %EXIT_CODE%
