@echo off
setlocal
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

rem The normal launch uses --no-sync. Install the configured speaker dependencies
rem once when diarization is enabled and its module is missing.
uv run --no-sync python -c "from pathlib import Path; from importlib.util import find_spec; from auto_translate.config import load_config; import sys; sys.exit(load_config(Path('config.toml')).mpv.speaker_diarization and not (find_spec('pyannote') and find_spec('pyannote.audio')))"
if errorlevel 1 (
    echo Installing diarization dependencies...
    uv sync --inexact --extra diarization
    if errorlevel 1 (
        echo ERROR: Could not install diarization dependencies.
        pause
        exit /b 1
    )
)

set "VIDEO_URL="
set /p "VIDEO_URL=Paste YouTube VOD URL: "

if not defined VIDEO_URL (
    echo No URL entered.
    pause
    exit /b 1
)

if not exist "logs" mkdir "logs"

echo.
echo Starting translation with MPV (paused)...
echo.

rem Run a single translator process. The previous two-uv pipe could exit
rem early and hide MPV startup errors; the CLI now prints JP/EN directly.
uv run --no-sync auto-translate "%VIDEO_URL%" --mode vod --mpv --no-web --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: Translator exited with code %EXIT_CODE%.
    echo Relevant errors from logs\run.log:
    findstr /i /c:"ERROR" /c:"Traceback" /c:"Exception" /c:"BrokenPipeError" "logs\run.log"
) else (
    echo Translator stopped.
)
echo Full diagnostic log: "%CD%\logs\run.log"
pause
exit /b %EXIT_CODE%
