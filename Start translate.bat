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
set /p "VIDEO_URL=Paste YouTube live or VOD URL: "

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
uv run --no-sync auto-translate "%VIDEO_URL%" --mode auto --mpv --delay 10 --no-web --simple-terminal 2>"logs\run.log"
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
