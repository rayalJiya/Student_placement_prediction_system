@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" phase5b_feature_engineering.py
if errorlevel 1 (
  echo Phase 5B failed. Copy the error message and send it to ChatGPT.
) else (
  echo Phase 5B completed. Open reports\phase5b_feature_engineering.md
)
pause
