@echo off
REM ═════════════════════════════════════════════════════════════════════════════
REM  TorusFold — one-click start.
REM
REM  Does three things, in order:
REM    1. finds a Python that can actually run the pipeline
REM    2. runs the external-dependency discovery and writes activate_deps.bat
REM    3. starts the server in this window, unbuffered, so a prediction scrolls
REM       past live as it runs
REM
REM  Everything it writes is machine-specific and git-ignorable:
REM    .env.local          the discovered paths, one per line
REM    activate_deps.bat   the same paths as `set` lines
REM
REM  Safe to run repeatedly. Ctrl+C stops the server.
REM ═════════════════════════════════════════════════════════════════════════════
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8877"

REM ── 1. Find an interpreter ───────────────────────────────────────────────────
REM The order is: an explicit choice, then whatever the setup pass discovered and
REM recorded in activate_deps.bat, then a conventional location. GPU-capable
REM interpreters come first among the fallbacks because RNAbpFlow is launched with
REM `--device cuda`, so the interpreter decides whether that predictor joins the
REM ensemble at all.
if exist "activate_deps.bat" call "activate_deps.bat" >nul 2>&1

set "PY="
if defined TORUSFOLD_PYTHON if exist "%TORUSFOLD_PYTHON%" set "PY=%TORUSFOLD_PYTHON%"
if not defined PY (
  for %%P in (
    "C:\ana\envs\comfyui\python.exe"
    "C:\ana\envs\circrna3d\python.exe"
    "C:\ana\envs\bio\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "C:\ProgramData\miniconda3\python.exe"
  ) do (
    if not defined PY if exist %%P set "PY=%%~P"
  )
)
if not defined PY (
  for %%P in (python.exe) do if not defined PY set "PY=%%~$PATH:P"
)
if not defined PY (
  echo.
  echo   ERROR: no Python found.
  echo   Set TORUSFOLD_PYTHON to the interpreter that has numpy and ViennaRNA,
  echo   for example:   set TORUSFOLD_PYTHON=C:\path\to\python.exe
  echo.
  pause
  exit /b 1
)

REM ViennaRNA, numpy and OpenMM are hard requirements — check before starting.
"%PY%" -c "import numpy, RNA, openmm" 2>nul
if errorlevel 1 (
  echo.
  echo   ERROR: %PY% cannot import numpy, ViennaRNA and OpenMM together.
  echo   Install what is missing, or point TORUSFOLD_PYTHON at another
  echo   interpreter:
  echo     conda install -c conda-forge viennarna openmm
  echo.
  pause
  exit /b 1
)

echo.
echo   TorusFold
echo     interpreter : %PY%
echo     url         : http://127.0.0.1:%PORT%/
echo.

REM ── 2. Discover and configure the external tools ─────────────────────────────
REM Writes .env.local + activate_deps.bat from what is actually on this machine.
REM Ten minutes of searching is the cost of not guessing; --fast skips the disk
REM search and only checks the locations already known.
set "DEPS_ARGS="
if /I "%~2"=="--fast" set "DEPS_ARGS=--roots ."
echo   Scanning for external tools ...
"%PY%" tools\configure_deps.py write %DEPS_ARGS%

if exist "activate_deps.bat" (
  call "activate_deps.bat" >nul
  echo   External tools configured from activate_deps.bat
) else (
  echo   No external tools were configured; the pipeline will run with the
  echo   predictors it can reach and skip the stages it cannot.
)

REM ── 3. Start the server ──────────────────────────────────────────────────────
echo.
echo   Starting the server. Prediction output scrolls below as it runs.
echo   Press Ctrl+C here to stop.
echo   ----------------------------------------------------------------------
echo.

"%PY%" -u serve.py %PORT%

echo.
echo   Server stopped.
pause
