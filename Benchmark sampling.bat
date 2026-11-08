@echo off
setlocal
chcp 65001 >nul
title Qwen3.5 Translation Sampling Benchmark
cd /d "%~dp0"
if not exist "logs" mkdir "logs"
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)
echo Comparing temperatures 0.1, 0.35, 0.5 and 0.7 on the same Japanese text.
echo Close Start translate.bat before running: benchmark owns the llama-server.
uv run --no-sync python tools\benchmark_qwen_sampling.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo ERROR: Benchmark failed with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
