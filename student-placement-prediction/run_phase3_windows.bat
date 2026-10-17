@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Please run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" phase3_train_select.py
if errorlevel 1 (
  echo Phase 3 failed. Copy the full error message and send it to ChatGPT.
) else (
  echo Phase 3 completed. Open reports\phase3_model_selection.md
)
pause
