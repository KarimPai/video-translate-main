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

set "VIDEO_URL="
set /p "VIDEO_URL=Paste YouTube VOD URL: "

if not defined VIDEO_URL (
    echo No URL entered.
    pause
    exit /b 1
)

echo.
echo Starting translation with MPV (paused)...
echo.

if not exist logs mkdir logs
rem TerminalOutput writes timed Japanese lines and indented English lines to stdout.
uv run --no-sync auto-translate "%VIDEO_URL%" --mode vod --mpv --no-web 2>"logs\run.log" | uv run --no-sync python -u -X utf8 -c "import re,sys; from collections import deque; p=re.compile(r'^\[[\d:.]+\u2013[\d:.]+\] (.*?)    \(vad |^\s*\u2192 (.+)$'); deque((print('JP: '+m[1] if m[1] is not None else 'EN: '+m[2]) for line in sys.stdin if (m:=p.match(line.rstrip()))),maxlen=0)"

echo.
echo Translator stopped.
pause