@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Voxtral + PLaMo - First-time Setup

echo ============================================
echo   Voxtral ASR + PLaMo 3 - Windows Setup
echo ============================================
echo.
echo This installs Windows tools, Python packages,
echo CUDA audio.cpp and CUDA llama.cpp.
echo Setup downloads Voxtral Q4_K and PLaMo Q4_K_M.
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
    echo GUI dependencies are installed by Setup.
) else (
    echo SETUP FAILED. Read the error above, then rerun Setup.bat.
)
echo.
pause
exit /b %SETUP_RESULT%
