@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
title Japanese Captions Translator
cd /d "%~dp0"

if errorlevel 1 (
    echo ERROR: Could not enter the project directory.
    pause
    exit /b 1
)

echo.
echo ========================================
echo       YouTube Caption Translator
echo ========================================
echo.

set "VIDEO_URL="
set /p "VIDEO_URL=Paste YouTube VOD URL: "

if not defined VIDEO_URL (
    echo No URL entered.
    pause
    exit /b 1
)

if not exist "logs" mkdir "logs"

echo.
echo Starting YouTube captions-only translation with PLaMo 8B...
echo.

rem Load this checkout even when the project's generated console launcher is stale.
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)
rem No VAD, ASR, diarization or DeepFilterNet dependencies are loaded.
uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --no-web --simple-terminal 2>"logs\run.log"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: Translator exited with code %EXIT_CODE%.
    echo Last startup messages:
    powershell -NoProfile -Command "Get-Content -LiteralPath 'logs\run.log' -Encoding UTF8 -Tail 6 | ForEach-Object { if ($_.Length -gt 350) { $_.Substring(0,350) } else { $_ } }"
) else (
    echo Translator stopped.
)
echo Full diagnostic log: "%CD%\logs\run.log"
pause
exit /b %EXIT_CODE%
