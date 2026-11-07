@echo off
setlocal
title Japanese ASR Comparison
cd /d "%~dp0"
if errorlevel 1 (
  echo ERROR: Unable to open test project folder.
  pause
  exit /b 1
)
echo.
echo ===========================================
echo   Japanese ASR Comparison - Test Only
echo   Haas / Qwen3 base / Japanese fine-tune
echo ===========================================
echo.
set "VIDEO_URL=https://www.youtube.com/watch?v=LE8iGHrakAc"
set "PREFERRED_URL="
set /p "PREFERRED_URL=YouTube URL [Enter for supplied sample]: "
if defined PREFERRED_URL set "VIDEO_URL=%PREFERRED_URL%"
echo.
echo Processing first 120 seconds of:
echo %VIDEO_URL%
echo.
echo First run may download the Japanese fine-tune model (about 2.18 GB).
echo This does NOT change the normal Start translate.bat.
echo.
uv run --no-sync python tools\compare_asr.py "%VIDEO_URL%" --seconds 120
set "EXITCODE=%ERRORLEVEL%"
echo.
if not "%EXITCODE%"=="0" (
  echo Benchmark incomplete. Open compare_finetune.txt and check Status / ERROR lines.
) else (
  echo Benchmark completed. Open compare_finetune.txt in the video's benchmark folder.
)
echo.
echo Default sample report:
echo "%~dp0logs\asr_benchmark\LE8iGHrakAc\compare_finetune.txt"
echo For a different video, use the absolute path printed by Python.
echo.
pause
exit /b %EXITCODE%
