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
echo   Haas / Haas vocabulary / Qwen3 1.7B
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
echo First run may download models; the video is downloaded only once.
echo This does NOT change the normal Start translate.bat.
echo.
uv run --no-sync python tools\compare_asr.py "%VIDEO_URL%" --seconds 120
set "EXITCODE=%ERRORLEVEL%"
echo.
if not "%EXITCODE%"=="0" (
  echo The comparison ended with an error, see the error above.
) else (
  echo Results: logs\asr_benchmark\VIDEO_ID\compare.txt
  echo Search within the folder using your YouTube ID.
)
echo.
pause
exit /b %EXITCODE%
