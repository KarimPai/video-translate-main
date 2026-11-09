@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title Granite Speech 5 - Japanese to English
cd /d "%~dp0"

if errorlevel 1 (
    echo ERROR: Cannot enter the project directory.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Granite Speech 5 Direct Translator
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

rem FireRedVAD remains required. GTCRN and Sherpa are disabled by default.
set "NEED_SYNC=0"
if not exist "uv.lock" set "NEED_SYNC=1"
uv run --no-sync python -c "import numpy, fireredvad, auto_translate" >nul 2>nul
if errorlevel 1 set "NEED_SYNC=1"
if "%NEED_SYNC%"=="1" (
    echo Installing Granite Speech and FireRedVAD runtime...
    uv sync
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)

echo.
echo Starting direct Granite 5 audio-to-English with MPV...
echo.

uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --delay 10 --no-web --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: Granite Speech translator exited with code %EXIT_CODE%.
    echo Last diagnostic messages:
    powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\run.log' -Encoding UTF8 -Tail 9 | ForEach-Object { if ($_.Length -gt 350) { $_.Substring(0,350) } else { $_ } }"
) else (
    echo Granite Speech translator stopped.
)
echo Application log: "%CD%\logs\run.log"
echo Model runtime:    "%CD%\logs\granite5-server.log"
pause
exit /b %EXIT_CODE%
