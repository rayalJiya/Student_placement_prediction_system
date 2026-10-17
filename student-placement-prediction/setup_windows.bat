@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Student Placement Prediction System - Setup
echo ============================================================
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
  echo During installation, tick "Add python.exe to PATH", then reopen this window.
  pause
  exit /b 1
)
python --version
if not exist ".venv\Scripts\python.exe" (
  echo Creating a project-local virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)
echo Installing project dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Setup complete. Next, double-click run_phase2_windows.bat
pause
exit /b 0
:error
echo Setup failed. Copy the error message and share it with ChatGPT for help.
pause
exit /b 1
