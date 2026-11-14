@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
rem Prefer the CUDA llama.cpp binary installed by Setup.bat.
set "LLAMA_PATH_FILE=%LOCALAPPDATA%\HaasPLaMo\dual-llama-bin-path.txt"
if exist "%LLAMA_PATH_FILE%" (
    for /f "usebackq delims=" %%P in ("%LLAMA_PATH_FILE%") do (
        if exist "%%P\llama-server.exe" set "PATH=%%P;%PATH%"
    )
)
title Haas + Whisper Turbo + LFM + PLaMo
cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot enter the project directory.
    pause
    exit /b 1
)
echo.
echo ========================================
echo   Haas + Whisper Turbo Japanese Dual ASR
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
echo YouTube video quality: up to %VIDEO_QUALITY%p (lower if unavailable)

set "RTX_VSR_ARG="
if not "%VIDEO_QUALITY%"=="480" goto vsr_choice_complete
echo.
echo NVIDIA App ^> System ^> Video: enable RTX Video Super Resolution and select High quality.
set "RTX_VSR_CHOICE="
set /p "RTX_VSR_CHOICE=Enable optional RTX VSR 480p-to-1440p enhancement? [y/N]: "
if /I "%RTX_VSR_CHOICE%"=="Y" set "RTX_VSR_ARG=--rtx-vsr"
:vsr_choice_complete
if defined RTX_VSR_ARG (
    echo RTX VSR requested: 3x scaling; NVIDIA App controls enhancement quality.
) else (
    echo RTX VSR off ^(standard MPV playback^).
)

if not exist "config.toml" copy /y "config.dual-asr.example.toml" "config.toml" >nul
if exist "%LOCALAPPDATA%\Programs\speech.cpp\current\speech.exe" set "PATH=%LOCALAPPDATA%\Programs\speech.cpp\current;%PATH%"
if not exist "logs" mkdir "logs"
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)
uv run --no-sync python -c "import numpy, fireredvad, faster_whisper, auto_translate" >nul 2>nul
if errorlevel 1 (
    echo Installing Haas + PLaMo pipeline Python dependencies...
    uv sync
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)
echo.
echo Starting Haas + Whisper Turbo ASR, LFM reconciler, PLaMo 3 Translate...
echo Original 16 kHz audio, FireRedVAD, Haas + budget-admitted Whisper Turbo ASR, LFM reconciliation, PLaMo translation.
echo Active models, VRAM readings and fallback reasons: logs\run.log
echo Whisper Turbo model downloads automatically on first launch ^(~1.6 GB^).
echo.

uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --shared-gpu-playback --no-web --hwdec auto-safe --video-quality %VIDEO_QUALITY% %RTX_VSR_ARG% --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: Translator exited with code %EXIT_CODE%.
    echo Last 45 application log lines:
    powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\run.log' -Encoding UTF8 -Tail 45"
    if exist "logs\mpv.log" (
        echo Last 30 MPV log lines:
        powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\mpv.log' -Encoding UTF8 -Tail 30"
    )
) else (
    echo Translator stopped.
)
echo Application log: "%CD%\logs\run.log"
pause
exit /b %EXIT_CODE%
