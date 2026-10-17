@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Please run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" phase5a_diagnose_compare.py
if errorlevel 1 (
  echo Phase 5A failed. Copy the full error message and send it to ChatGPT.
) else (
  echo Phase 5A completed. Open reports\phase5a_diagnostics.md
)
pause
