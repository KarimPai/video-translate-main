@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot enter project folder.
    pause
    exit /b 1
)
where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv not installed. Run Setup.bat first.
    pause
    exit /b 1
)
echo Installing/verifying GUI requirements (first launch downloads PySide6)...
uv sync --extra gui
if errorlevel 1 (
    echo ERROR: GUI environment setup failed. Run Setup.bat, then retry.
    pause
    exit /b 1
)
set "PYTHONPATH=%CD%\src"
".venv\Scripts\python.exe" -m auto_translate.gui
if errorlevel 1 (
    echo ERROR: GUI exited with an error.
    pause
    exit /b 1
)
endlocal
