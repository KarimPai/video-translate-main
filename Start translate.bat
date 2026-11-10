@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title Anime Whisper ASR + PLaMo 3 Translate 2B
cd /d "%~dp0"

if errorlevel 1 (
    echo ERROR: Cannot enter the project directory.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Anime Whisper ASR + PLaMo 3 2B
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
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)

rem Stability baseline disables optional GTCRN and Sherpa segmentation.
rem Do not import sherpa-onnx in preflight; its native ORT DLLs are unnecessary.
set "NEED_SYNC=0"
if not exist "uv.lock" set "NEED_SYNC=1"
uv run --no-sync python -c "import numpy, fireredvad, torch, torchaudio, auto_translate; from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor" >nul 2>nul
if errorlevel 1 set "NEED_SYNC=1"
if "%NEED_SYNC%"=="1" (
    echo Installing Anime Whisper CUDA and FireRedVAD dependencies...
    uv sync --no-progress
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        echo Try: uv sync
        pause
        exit /b 1
    )
)

echo.
echo Starting Anime Whisper Japanese transcription + PLaMo English translation...
echo Accuracy test: no ASR initial prompts or hotwords, MPV software decoding,
echo                  PLaMo limited GPU offload (16 layers).
echo.

uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --delay 10 --hwdec no --plamo-gpu-layers 16 --audio-profile baseline --diagnostics --no-web --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: Anime Whisper + PLaMo translator exited with code %EXIT_CODE%.
    echo Last 45 lines of application log:
    powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\run.log' -Encoding UTF8 -Tail 45"
    echo Last 30 lines of MPV log:
    if exist "logs\mpv.log" powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\mpv.log' -Encoding UTF8 -Tail 30"
) else (
    echo Anime Whisper + PLaMo translator stopped.
)
echo Application log: "%CD%\logs\run.log"
pause
exit /b %EXIT_CODE%
