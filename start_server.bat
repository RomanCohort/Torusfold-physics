@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  TorusFold web server, started so you can WATCH it.
REM
REM  Run this by double-clicking it, or from a terminal. It opens a console
REM  window that shows the pipeline's own output live while a prediction runs.
REM
REM  Two details that matter:
REM
REM   * -u  (unbuffered). Python block-buffers stdout when it is not a terminal,
REM     so without this the window stays blank for minutes and then dumps
REM     everything at once. The server also mirrors its output to a per-job log
REM     file, but the console needs -u to be live as well.
REM
REM   * the interpreter is given by full path, because this machine has no
REM     `python` on PATH.
REM
REM  To see only the pipeline and skip the HTTP request lines, use watch_log.bat,
REM  or filter this window with:  ... | findstr /v "HTTP/1.1"
REM ─────────────────────────────────────────────────────────────────────────────
setlocal

set "PY=C:\ana\envs\circrna3d\python.exe"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8877"

cd /d "%~dp0"

if not exist "%PY%" (
  echo.
  echo   ERROR: interpreter not found at
  echo     %PY%
  echo   Edit the PY line in this file to point at a Python that has
  echo   numpy and ViennaRNA installed.
  echo.
  pause
  exit /b 1
)

echo.
echo   TorusFold web server
echo     interpreter : %PY%
echo     url         : http://127.0.0.1:%PORT%/
echo     live log    : output_web\logs\^<job^>.log
echo.
echo   Prediction output will scroll below as it runs.
echo   Press Ctrl+C here to stop the server.
echo.

"%PY%" -u serve.py %PORT%

echo.
echo   Server stopped.
pause
