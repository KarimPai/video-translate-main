@echo off
title Japanese Video Translator
cd /d "C:\Games\video-translate-main"

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