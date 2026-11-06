@echo off
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

set /p VIDEO_URL=Paste YouTube VOD URL:

if "%VIDEO_URL%"=="" (
    echo No URL entered.
    pause
    exit /b 1
)

echo.
echo Starting translation with MPV (paused)...
echo.

uv run --no-sync auto-translate "%VIDEO_URL%" --mode vod --mpv --no-web

 echo.
echo Translator stopped.
pause