@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title Whisper Large-v3 Direct AST
cd /d "%~dp0"

if errorlevel 1 (
    echo ERROR: Cannot enter the project directory.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Whisper Large-v3 Direct JA to EN AST
echo ========================================
echo.

set "VIDEO_URL="
set /p "VIDEO_URL=Paste YouTube live or VOD URL: "
if not defined VIDEO_URL (
    echo No URL entered.
    pause
    exit /b 1
)

:select_quality
set "VIDEO_QUALITY="
echo.
set /p "VIDEO_QUALITY=YouTube video quality (360/480/720/1080) [720]: "
if not defined VIDEO_QUALITY set "VIDEO_QUALITY=720"
if "%VIDEO_QUALITY%"=="360" goto quality_selected
if "%VIDEO_QUALITY%"=="480" goto quality_selected
if "%VIDEO_QUALITY%"=="720" goto quality_selected
if "%VIDEO_QUALITY%"=="1080" goto quality_selected
echo Invalid quality. Enter 360, 480, 720, or 1080.
goto select_quality

:quality_selected
echo YouTube video quality: up to %VIDEO_QUALITY%p (lower if unavailable)

if not exist "logs" mkdir "logs"
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)

rem Stability baseline disables optional GTCRN and Sherpa segmentation.
rem Do not import sherpa-onnx in preflight; its native ORT DLLs are unnecessary.
set "NEED_SYNC=0"
if not exist "uv.lock" set "NEED_SYNC=1"
uv run --no-sync python -c "import numpy, fireredvad, faster_whisper, auto_translate" >nul 2>nul
if errorlevel 1 set "NEED_SYNC=1"
if "%NEED_SYNC%"=="1" (
    echo Installing Whisper Large-v3 and FireRedVAD dependencies...
    uv sync
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        echo Try: uv sync
        pause
        exit /b 1
    )
)

echo.
echo Starting Whisper direct Japanese-audio-to-English translation...
echo Playback hitch test: optional audio processors OFF, MPV safe hardware decoding,
echo                  Whisper efficient mixed precision (stable GPU decode baseline).
echo.

uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --delay 10 --hwdec auto-safe --video-quality %VIDEO_QUALITY% --whisper-compute int8_float16 --audio-profile baseline --no-web --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: Whisper direct AST exited with code %EXIT_CODE%.
    echo Last 45 lines of application log:
    powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\run.log' -Encoding UTF8 -Tail 45"
    echo Last 30 lines of MPV log:
    if exist "logs\mpv.log" powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\mpv.log' -Encoding UTF8 -Tail 30"
) else (
    echo Whisper direct AST stopped.
)
echo Application log: "%CD%\logs\run.log"
pause
exit /b %EXIT_CODE%
