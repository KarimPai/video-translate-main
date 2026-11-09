@echo off
setlocal
chcp 65001 >nul
title ReazonSpeech k2 v2 - PLaMo 3 Translate 2B
cd /d "%~dp0"

if errorlevel 1 (
    echo ERROR: Could not enter the project directory.
    pause
    exit /b 1
)

echo.
echo ========================================
echo       ReazonSpeech + PLaMo 3 2B
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
echo Initializing ReazonSpeech k2 v2 and PLaMo 3 Translate 2B...
echo.
rem The branch adds sherpa-onnx; uv.lock may need regeneration on first run.
uv run --no-sync python -c "from importlib.metadata import version; version('sherpa-onnx'); version('sherpa-onnx-core'); import huggingface_hub, fireredvad, auto_translate" >nul 2>nul
if errorlevel 1 (
    echo Installing missing dependencies and updating the local uv lock...
    uv sync --no-progress
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        echo Try: uv sync
        pause
        exit /b 1
    )
)
echo.
echo Starting JP to EN translation with MPV...

rem Run a single translator process. The previous two-uv pipe could exit
rem early and hide MPV startup errors; the CLI now prints JP/EN directly.
uv run --no-sync auto-translate "%VIDEO_URL%" --mode auto --mpv --no-web --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: Translator exited with code %EXIT_CODE%.
    echo Last 35 lines from logs\run.log:
    powershell -NoProfile -Command "Get-Content 'logs\run.log' -Tail 35"
    if "%EXIT_CODE%"=="-1073741819" echo Native access violation 0xC0000005: check sherpa worker and ONNX Runtime DLL conflicts.
) else (
    echo Translator stopped.
)
echo Full diagnostic log: "%CD%\logs\run.log"
pause
exit /b %EXIT_CODE%
