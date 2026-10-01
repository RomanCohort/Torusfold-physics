@echo off
REM ============================================================================
REM  TorusFold - start
REM
REM  Double-click this. It finds an interpreter, configures the external tools if
REM  that has not been done yet, starts the server in THIS window, and opens the
REM  address. Prediction output scrolls past live as it runs.
REM
REM  Usage:
REM    start.bat                 start on port 8877
REM    start.bat 8899            start on another port
REM    start.bat --setup         re-run the tool discovery first, full search
REM    start.bat --check         report the environment and exit, start nothing
REM
REM  Why the window shows live output: the server is started with -u. Python
REM  block-buffers stdout when it is not a terminal, so without -u the window sits
REM  blank for minutes and then dumps everything at once.
REM
REM  Related:
REM    watch_log.bat         follow the current run's log in a second terminal
REM    start_server.bat      start the server only, without the setup pass
REM
REM  NOTE: this file is deliberately ASCII-only. A .bat is read as bytes in the
REM  console's active code page, so non-ASCII characters here are parsed as stray
REM  commands ("'g' is not recognized as an internal or external command").
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PORT=8877"
set "MODE=start"
for %%A in (%*) do (
  if /I "%%A"=="--setup" set "MODE=setup"
  if /I "%%A"=="--check" set "MODE=check"
  echo %%A| findstr /r "^[0-9][0-9]*$" >nul && set "PORT=%%A"
)

REM -- 1. An interpreter -------------------------------------------------------
REM activate_deps.bat records what the last setup found, including
REM TORUSFOLD_PYTHON. Loading it first means this uses the interpreter that was
REM verified to import the pipeline, rather than guessing again.
if exist "activate_deps.bat" call "activate_deps.bat" >nul 2>&1

set "PY="
if defined TORUSFOLD_PYTHON if exist "%TORUSFOLD_PYTHON%" set "PY=%TORUSFOLD_PYTHON%"
if not defined PY (
  REM python.org installs first: the pipeline does not need conda, and a machine
  REM with a normal Python installation was previously told to install Anaconda.
  for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python314\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\ana\envs\comfyui\python.exe"
    "C:\ana\envs\circrna3d\python.exe"
    "C:\ana\envs\bio\python.exe"
    "C:\anaconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\miniconda3\python.exe"
    "C:\ProgramData\miniconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniforge3\python.exe"
  ) do (
    if not defined PY if exist %%P set "PY=%%~P"
  )
)
if not defined PY (
  for %%P in (python.exe) do if not defined PY set "PY=%%~$PATH:P"
)

if not defined PY (
  echo.
  echo   No Python found.
  echo.
  echo   Python 3.12 or newer is needed. Anaconda is NOT required: every
  echo   package the pipeline imports has a Windows installer on PyPI.
  echo.
  echo   1. Install Python from  https://www.python.org/downloads/
  echo      Tick "Add python.exe to PATH" in the installer.
  echo   2. Open a NEW terminal and run:
  echo        python -m pip install numpy scipy ViennaRNA openmm matplotlib
  echo        python -m pip install ml_collections biopython dm-tree einops
  echo        python -m pip install gemmi freesasa pandas transformers torch
  echo      For an AMD GPU use the ROCm build of torch from pytorch.org.
  echo   3. Run this launcher again.
  echo.
  echo   If you already have an interpreter somewhere else:
  echo       set TORUSFOLD_PYTHON=C:\path\to\python.exe
  echo   Or with conda, which also works:
  echo       conda create -n torusfold -c conda-forge python=3.12 numpy scipy openmm pytorch
  echo       conda activate torusfold
  echo       pip install ViennaRNA dm-tree freesasa transformers
  echo.
  pause
  exit /b 1
)

REM The three imports that decide whether the pipeline can start at all. Checked
REM before anything else, so a wrong interpreter is named now rather than 40
REM seconds into a run.
"%PY%" -c "import numpy, RNA, openmm" 2>nul
if errorlevel 1 (
  echo.
  echo   %PY%
  echo   cannot import numpy, ViennaRNA and OpenMM together.
  echo.
  if exist "tools\install_deps.py" (
    "%PY%" tools\install_deps.py --env-only
    echo.
  )
  echo   Add what is missing, or set TORUSFOLD_PYTHON to another interpreter.
  echo.
  pause
  exit /b 1
)

if not exist "tools\configure_deps.py" (
  echo.
  echo   tools\configure_deps.py is missing. Is this a full checkout?
  echo.
  pause
  exit /b 1
)

REM -- 2. Configure the external tools, once -----------------------------------
if /I "%MODE%"=="setup" (
  echo.
  echo   Surveying interpreters and external tools. This searches the drive and
  echo   can take a few minutes. It is a one-time step.
  echo.
  "%PY%" tools\install_deps.py --env-only
  echo.
  "%PY%" tools\configure_deps.py write
  goto :launch
)

if /I "%MODE%"=="check" (
  echo.
  "%PY%" tools\install_deps.py --env-only
  echo.
  "%PY%" tools\configure_deps.py check
  echo.
  echo   Nothing was started. Run  start.bat  to launch the server.
  echo.
  pause
  exit /b 0
)

if not exist "activate_deps.bat" (
  echo.
  echo   First run: looking for the external tools beside this repository.
  echo   If that finds nothing, run  start.bat --setup  for a full drive search.
  echo.
  "%PY%" tools\configure_deps.py write --fast
  if exist "activate_deps.bat" call "activate_deps.bat" >nul 2>&1
  echo.
)

:launch
echo.
echo   ----------------------------------------------------------------------
echo    TorusFold
echo      interpreter : %PY%
echo      open this   : http://127.0.0.1:%PORT%/
echo      live log    : output_web\logs\JOB.log
echo   ----------------------------------------------------------------------
echo    Prediction output appears below as it runs.
echo    Press Ctrl+C here to stop the server.
echo.

REM Open the address once the server has had time to bind. Run detached so the
REM server stays in the foreground and Ctrl+C still reaches it.
start "" /min cmd /c "timeout /t 8 /nobreak >nul & start "" http://127.0.0.1:%PORT%/"

"%PY%" -u serve.py %PORT%

echo.
echo   Server stopped.
pause
