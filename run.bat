@echo off
rem ============================================================
rem  COF Studio - start the local web UI
rem  Keep this file ASCII-only: cmd.exe reads .bat in the OEM
rem  codepage, so Chinese text here would come out as garbage.
rem ============================================================
setlocal
cd /d "%~dp0"
title COF Studio

set "VPY="

rem ---------- 1. bundled portable runtime: nothing to install ----------
if exist "runtime\python.exe" (
  "runtime\python.exe" -c "import numpy,ase,scipy,rdkit" >nul 2>&1
  if not errorlevel 1 set "VPY=runtime\python.exe"
)

rem ---------- 2. local virtual environment ----------
if not defined VPY if exist ".venv\Scripts\python.exe" set "VPY=.venv\Scripts\python.exe"

rem ---------- 3. system python ----------
if not defined VPY (
  where py >nul 2>&1 && set "VPY=py -3"
)
if not defined VPY (
  where python >nul 2>&1 && set "VPY=python"
)

if not defined VPY (
  echo.
  echo   [ERROR] Python not found, and the portable "runtime" folder is missing.
  echo.
  echo   If you got this folder from someone else, you probably copied only
  echo   some of the files. Ask for the complete .zip and unzip all of it,
  echo   keeping the "runtime" sub-folder next to run.bat.
  echo.
  echo   Or install Python 3.10+ from python.org, tick "Add to PATH",
  echo   then double-click install.bat and run this again.
  echo.
  pause
  exit /b 1
)

echo.
echo   COF Studio starting ...
echo   Interpreter: %VPY%
echo   The browser will open automatically.
echo   Close this window to stop the program.
echo.

%VPY% cof_studio.py --open
if errorlevel 1 (
  echo.
  echo   Program exited with an error - read the messages above.
  echo   If a module is missing, double-click install.bat.
  echo.
  pause
)
