@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv is missing. Run Setup.bat first.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Backend environment missing. Run Setup.bat first.
    goto :failed
)
echo Verifying GUI dependencies without resyncing existing backend models...
".venv\Scripts\python.exe" -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    uv pip install --python ".venv\Scripts\python.exe" "PySide6>=6.8,<7"
    if errorlevel 1 goto :failed
)
echo Installing PyInstaller into existing virtual environment...
uv pip install --python ".venv\Scripts\python.exe" "pyinstaller>=6,<7"
if errorlevel 1 goto :failed
echo Building native Windows GUI launcher (models and backend remain external)...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onedir --windowed --name VideoTranslate --paths src --distpath dist --workpath build src\auto_translate\gui.py
if errorlevel 1 goto :failed
echo.
echo Build complete: "%CD%\dist\VideoTranslate\VideoTranslate.exe"
echo IMPORTANT: Keep this executable in the project dist folder.
echo It uses project .venv, mpv, CUDA/llama and existing model downloads.
pause
exit /b 0
:failed
echo GUI build failed. See the error above.
pause
exit /b 1
