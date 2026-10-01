@echo off
REM ============================================================================
REM  TorusFold web server, started so you can WATCH it.
REM
REM  Equivalent to start.bat; kept as a separate entry point because its name is
REM  the one people reach for when they want the server rather than the launcher.
REM
REM  Two details that matter:
REM
REM   * -u  (unbuffered). Python block-buffers stdout when it is not a terminal,
REM     so without this the window stays blank for minutes and then dumps
REM     everything at once. The server also mirrors its output to a per-job log
REM     file, but the console needs -u to be live as well.
REM
REM   * the interpreter is resolved from activate_deps.bat first, then by full
REM     path, because this machine has no `python` on PATH.
REM
REM  NOTE: ASCII-only on purpose. A .bat is read as bytes in the console's active
REM  code page, so non-ASCII characters here are parsed as stray commands.
REM ============================================================================
setlocal

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8877"

cd /d "%~dp0"

if exist "activate_deps.bat" call "activate_deps.bat" >nul 2>&1

set "PY="
if defined TORUSFOLD_PYTHON if exist "%TORUSFOLD_PYTHON%" set "PY=%TORUSFOLD_PYTHON%"
if not defined PY (
  for %%P in (
    "C:\ana\envs\comfyui\python.exe"
    "C:\ana\envs\circrna3d\python.exe"
    "C:\ana\envs\bio\python.exe"
  ) do (
    if not defined PY if exist %%P set "PY=%%~P"
  )
)

if not defined PY (
  echo.
  echo   ERROR: interpreter not found.
  echo   Set TORUSFOLD_PYTHON, or edit the candidate list in this file to point
  echo   at a Python that has numpy and ViennaRNA installed.
  echo.
  pause
  exit /b 1
)

echo.
echo   TorusFold web server
echo     interpreter : %PY%
echo     url         : http://127.0.0.1:%PORT%/
echo     live log    : output_web\logs\JOB.log
echo.
echo   Prediction output will scroll below as it runs.
echo   Press Ctrl+C here to stop the server.
echo.

"%PY%" -u serve.py %PORT%

echo.
echo   Server stopped.
pause
