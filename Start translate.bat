@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
rem Runtime paths are relative to this independent checkout.
set "LLAMA_PATH_FILE=%~dp0.cache\llama-bin-path.txt"
if exist "%LLAMA_PATH_FILE%" (
    for /f "usebackq delims=" %%P in ("%LLAMA_PATH_FILE%") do (
        if exist "%%P\llama-server.exe" set "PATH=%%P;%PATH%"
    )
)
title Qwen3-ASR + HY-MT
cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot enter the project directory.
    pause
    exit /b 1
)
echo.
echo ========================================
echo   Qwen3-ASR Japanese ASR + HY-MT
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
setlocal EnableDelayedExpansion
if not defined QUALITY_CHOICE set "QUALITY_CHOICE=3"
if "!QUALITY_CHOICE!"=="1" set "VIDEO_QUALITY=360"
if "!QUALITY_CHOICE!"=="2" set "VIDEO_QUALITY=480"
if "!QUALITY_CHOICE!"=="3" set "VIDEO_QUALITY=720"
if "!QUALITY_CHOICE!"=="4" set "VIDEO_QUALITY=1080"
if not defined VIDEO_QUALITY (
    echo Invalid choice. Enter 1, 2, 3, or 4.
    endlocal
    goto select_quality
)
echo YouTube video quality: up to %VIDEO_QUALITY%p (lower if unavailable)

if not exist "config.toml" copy /y "config.example.toml" "config.toml" >nul
if not exist "logs" mkdir "logs"
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)
uv run --no-sync python -c "import numpy, fireredvad, onnxruntime, auto_translate" >nul 2>nul
if errorlevel 1 (
    echo Installing Qwen3-ASR + HY-MT pipeline Python dependencies...
    uv sync --locked
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)
echo.
echo Starting persistent Qwen3-ASR-1.7B and Hy-MT2-7B on CUDA llama.cpp...
echo Original 16 kHz audio, CPU FireRedVAD, offline Qwen3-ASR utterances, English subtitles.
echo Runtime and VRAM diagnostics: logs\run.log
echo.

set "VIDEO_TRANSLATE_CLI_LOG=1"
uv run --no-sync python -m auto_translate --mode auto --mpv --shared-gpu-playback --no-web --hwdec auto-safe --video-quality %VIDEO_QUALITY% --simple-terminal
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
