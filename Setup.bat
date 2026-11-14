@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Haas + PLaMo - First-time Setup

echo ============================================
echo   Haas ASR + PLaMo 3 - Windows Setup
echo ============================================
echo.
echo This installs Windows tools, Python packages,
echo Vulkan speech.cpp, Haas and CUDA llama.cpp.
echo PLaMo Q4_K_M downloads on first translation.
echo Downloads and installers require internet access.
echo.

if not exist "%~dp0scripts\Setup-Windows.ps1" (
    echo ERROR: scripts\Setup-Windows.ps1 is missing.
    echo Extract the ENTIRE ZIP before running Setup.bat.
    pause
    exit /b 1
)

rem Appending dot prevents a trailing backslash from escaping the closing quote.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Setup-Windows.ps1" -ProjectDir "%~dp0."
set "SETUP_RESULT=%ERRORLEVEL%"
echo.
if "%SETUP_RESULT%"=="0" (
    echo SETUP FINISHED. Double-click "Start GUI.vbs" to launch the app.
    echo The first GUI launch installs PySide6 if necessary.
    echo First translation downloads PLaMo 3 and Whisper Large-v3 Turbo models.
) else (
    echo SETUP FAILED. Read the error above, then rerun Setup.bat.
)
echo.
pause
exit /b %SETUP_RESULT%
