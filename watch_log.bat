@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Follow the CURRENT run's log file in a terminal.
REM
REM  Use this when the server is already running (perhaps started from the web
REM  UI, or in another window) and you just want to watch a prediction. It finds
REM  the newest file in output_web\logs and follows it.
REM
REM  Why this works: the server mirrors everything the browser sees to a
REM  line-buffered log file, so it is written as the run happens rather than
REM  when some buffer happens to fill.
REM
REM  Ctrl+C stops following. It does NOT stop the server.
REM ─────────────────────────────────────────────────────────────────────────────
setlocal

cd /d "%~dp0"

if not exist "output_web\logs" (
  echo.
  echo   output_web\logs does not exist yet.
  echo   It is created when the first prediction starts.
  echo.
  pause
  exit /b 1
)

set "LATEST="
for /f "delims=" %%F in ('dir /b /o-d "output_web\logs\*.log" 2^>nul') do (
  if not defined LATEST set "LATEST=output_web\logs\%%F"
)

if not defined LATEST (
  echo.
  echo   No .log files in output_web\logs yet.
  echo   Start a prediction and run this again.
  echo.
  pause
  exit /b 1
)

echo.
echo   Following %LATEST%
echo   Ctrl+C to stop following. The server keeps running.
echo.

powershell -NoProfile -Command "Get-Content -LiteralPath '%LATEST%' -Wait -Tail 200"

echo.
pause
