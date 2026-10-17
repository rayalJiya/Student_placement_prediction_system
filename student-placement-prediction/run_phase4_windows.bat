@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Please run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" phase4_evaluate.py
if errorlevel 1 (
  echo.
  echo Phase 4 did not finish. Read the error above and share it for help.
  pause
  exit /b 1
)
echo.
echo Phase 4 evaluation and validation completed.
pause
