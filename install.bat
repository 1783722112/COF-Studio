@echo off
rem ============================================================
rem  COF Studio - one-click environment setup
rem  ASCII only: cmd.exe reads .bat in the OEM codepage.
rem ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
title COF Studio - setup

echo.
echo ============================================================
echo    COF Studio   Environment Setup
echo ============================================================
echo.

rem ---------- 1. locate Python ----------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo   [ERROR] Python not found.
  echo.
  echo   Install Python 3.10 or newer from
  echo     https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH", then run this again.
  echo.
  pause
  exit /b 1
)
echo   Interpreter: %PY%
%PY% -c "import sys; print('   Python', sys.version.split()[0])"
if errorlevel 1 (
  echo   [ERROR] Python 3.10+ required.
  pause
  exit /b 1
)

rem ---------- 2. virtual env ----------
echo.
echo ------------------------------------------------------------
echo   [1/3] Creating virtual environment .venv
echo ------------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
  echo   already exists, skipping
) else (
  %PY% -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo   [ERROR] failed to create virtual environment
    pause
    exit /b 1
  )
  echo   done
)
set "VPY=.venv\Scripts\python.exe"

echo   upgrading pip ...
"%VPY%" -m pip install --upgrade pip -q --disable-pip-version-check

rem ---------- 3. modelling deps ----------
echo.
echo ------------------------------------------------------------
echo   [2/3] Modelling deps:  numpy ase scipy matplotlib rdkit
echo ------------------------------------------------------------
"%VPY%" -m pip install numpy ase scipy matplotlib rdkit --disable-pip-version-check
if errorlevel 1 (
  echo.
  echo   [ERROR] install failed. If it is a network issue, try:
  echo     "%VPY%" -m pip install numpy ase scipy matplotlib rdkit -i https://pypi.tuna.tsinghua.edu.cn/simple
  echo.
  pause
  exit /b 1
)
echo   modelling environment ready

rem ---------- 4. optimisation deps (optional) ----------
echo.
echo ------------------------------------------------------------
echo   [3/3] Optimisation deps:  torch + mace-torch  (1-2 GB)
echo ------------------------------------------------------------
echo   Optional. Without it you can still build structures,
echo   you just cannot run the force-field relaxation.
echo.
set "ANS="
set /p ANS=   Install now? [y/N]: 
if /i not "%ANS%"=="y" goto skip_opt

echo.
echo   installing torch (CPU build, ~200 MB) ...
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
if errorlevel 1 (
  echo   [WARN] torch install failed, you can add it later
  goto skip_opt
)
echo.
echo   installing mace-torch + torch-dftd ...
"%VPY%" -m pip install mace-torch torch-dftd --disable-pip-version-check
if errorlevel 1 (
  echo   [WARN] mace-torch install failed, you can add it later
  goto skip_opt
)
echo   optimisation environment ready

:skip_opt

rem ---------- 5. self check ----------
echo.
echo ------------------------------------------------------------
echo   Self check
echo ------------------------------------------------------------
"%VPY%" -c "import numpy,ase,scipy,matplotlib;print('   build : numpy',numpy.__version__,'ase',ase.__version__,'OK')" 2>nul || echo   build : PROBLEM
"%VPY%" -c "import torch,mace;print('   relax : torch',torch.__version__,'mace OK')" 2>nul || echo   relax : not installed (build only)

echo.
echo ============================================================
echo    Done. Double-click run.bat to start.
echo ============================================================
echo.
pause
