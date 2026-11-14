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
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Backend environment missing. Run Setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    echo Installing only the PySide6 GUI package, preserving current backend dependencies...
    uv pip install --python ".venv\Scripts\python.exe" "PySide6>=6.8,<7"
    if errorlevel 1 (
        echo ERROR: GUI dependency install failed.
        pause
        exit /b 1
    )
)
echo Launching the GUI without a console window...
wscript.exe "%~dp0Start GUI.vbs"
if errorlevel 1 (
    echo ERROR: Could not launch GUI. Check the launcher and Python environment.
    pause
    exit /b 1
)
exit /b 0
