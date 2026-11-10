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

rem Sherpa-ONNX provides both CPU GTCRN enhancement and speaker-turn diarization.
rem Check the installed package version without importing native ORT DLLs during sync detection.
set "NEED_SYNC=0"
if not exist "uv.lock" set "NEED_SYNC=1"
uv run --no-sync python -c "import numpy, fireredvad, faster_whisper, auto_translate; import importlib.metadata as m; m.version('sherpa-onnx')" >nul 2>nul
if errorlevel 1 set "NEED_SYNC=1"
if "%NEED_SYNC%"=="1" (
    echo Installing Whisper Large-v3, FireRedVAD and Sherpa-ONNX CPU audio models...
    uv sync --extra diarization
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        echo Try: uv sync --extra diarization
        pause
        exit /b 1
    )
)
rem Check that the Sherpa native runtime loads before opening the player.
uv run --no-sync python -c "import sherpa_onnx" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Sherpa-ONNX failed to import. Check Windows native runtime/DLL dependencies.
    echo Try: uv sync --extra diarization
    pause
    exit /b 1
)

echo.
echo Starting Whisper direct Japanese-audio-to-English translation...
echo Audio processing: GTCRN enhancement ON, Sherpa speaker turns ON (CPU).
echo MPV safe hardware decoding, Whisper int8_float16 (stable GPU baseline).
echo.

uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --delay 10 --hwdec auto-safe --video-quality %VIDEO_QUALITY% --whisper-compute int8_float16 --audio-profile both --no-web --simple-terminal 2>"logs\run.log"
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
