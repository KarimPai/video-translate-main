@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title VoicePing Qwen3 Direct AST
cd /d "%~dp0"

if errorlevel 1 (
    echo ERROR: Cannot enter the project directory.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   VoicePing Qwen3 Direct JA to EN AST
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
set "QUALITY_CHOICE="
set "VIDEO_QUALITY="
echo.
echo Select YouTube video quality:
echo   1. 360p
echo   2. 480p
echo   3. 720p ^(default^)
echo   4. 1080p
set /p "QUALITY_CHOICE=Enter choice [3]: "
if not defined QUALITY_CHOICE set "QUALITY_CHOICE=3"
if "%QUALITY_CHOICE%"=="1" set "VIDEO_QUALITY=360"
if "%QUALITY_CHOICE%"=="2" set "VIDEO_QUALITY=480"
if "%QUALITY_CHOICE%"=="3" set "VIDEO_QUALITY=720"
if "%QUALITY_CHOICE%"=="4" set "VIDEO_QUALITY=1080"
if not defined VIDEO_QUALITY (
    echo Invalid choice. Enter 1, 2, 3, or 4.
    goto select_quality
)

:quality_selected
echo YouTube video quality: up to %VIDEO_QUALITY%p (lower if unavailable)

if not exist "logs" mkdir "logs"
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)

set "NEED_SYNC=0"
if not exist "uv.lock" set "NEED_SYNC=1"
uv run --no-sync python -c "import numpy, fireredvad, torch, transformers, qwen_asr, auto_translate" >nul 2>nul
if errorlevel 1 set "NEED_SYNC=1"
if "%NEED_SYNC%"=="1" (
    echo Installing VoicePing Qwen3 AST and FireRedVAD dependencies...
    uv sync
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        echo Try: uv sync
        pause
        exit /b 1
    )
)
echo.
echo Starting VoicePing direct Japanese-audio-to-English translation...
echo Audio processing: Original 16 kHz mono audio to FireRedVAD and VoicePing AST.
echo MPV safe hardware decoding, VoicePing BF16, no aligner or quantization.
echo.

uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --delay 10 --hwdec auto-safe --video-quality %VIDEO_QUALITY% --no-web --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: VoicePing AST exited with code %EXIT_CODE%.
    echo Last 45 lines of application log:
    powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\run.log' -Encoding UTF8 -Tail 45"
    echo Last 30 lines of MPV log:
    if exist "logs\mpv.log" powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\mpv.log' -Encoding UTF8 -Tail 30"
) else (
    echo VoicePing AST stopped.
)
echo Application log: "%CD%\logs\run.log"
pause
exit /b %EXIT_CODE%
