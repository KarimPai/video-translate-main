@echo off
setlocal DisableDelayedExpansion
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

rem Load this checkout even when the project's generated console launcher is stale.
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)
rem Keep normal --no-sync startup; install the DeepFilterNet3 extra only once
rem if the effective local configuration enables it and packages are missing.
uv run --no-sync python -c "import importlib.util; from pathlib import Path; from auto_translate.config import load_config; cfg=load_config(Path('config.toml')); assert not cfg.deepfilter.enabled or (importlib.util.find_spec('deepfilternet_rs') and importlib.util.find_spec('scipy'))" >nul 2>nul
if errorlevel 1 (
    echo Installing missing DeepFilterNet3 dependencies...
    uv sync --extra diarization --extra deepfilter
    if errorlevel 1 (
        echo ERROR: Could not install DeepFilterNet3 dependencies.
        pause
        exit /b 1
    )
)

uv run --no-sync python -m auto_translate "%VIDEO_URL%" --mode auto --mpv --delay 10 --no-web --simple-terminal 2>"logs\run.log"
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
