@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   COF Studio - build desktop exe
echo ============================================================
echo.

if not exist ".build_env\Scripts\python.exe" (
  echo [1/3] creating build environment .build_env ...
  py -3 -m venv .build_env 2>nul || python -m venv .build_env
  if not exist ".build_env\Scripts\python.exe" goto :no_python
  ".build_env\Scripts\python.exe" -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple
  ".build_env\Scripts\python.exe" -m pip install pyinstaller numpy ase scipy rdkit -i https://mirrors.aliyun.com/pypi/simple
  if errorlevel 1 goto :fail
) else (
  echo [1/3] build environment found, skip.
)

echo.
echo [2/3] packaging ... ^(about 1 minute^)
".build_env\Scripts\python.exe" -m PyInstaller --noconfirm --clean --distpath ".." --workpath ".build_tmp" cof_studio.spec
if errorlevel 1 goto :fail

echo.
echo [3/3] copying readme ...
if exist "..\COF Studio\" copy /y *.txt "..\COF Studio\" >nul

echo.
echo ============================================================
echo   DONE  -^>  ..\COF Studio\COFStudio.exe
echo ============================================================
pause
exit /b 0

:no_python
echo.
echo [ERROR] Python not found. Install Python 3.9+ first.
pause
exit /b 1

:fail
echo.
echo [ERROR] Build failed. See messages above.
pause
exit /b 1
