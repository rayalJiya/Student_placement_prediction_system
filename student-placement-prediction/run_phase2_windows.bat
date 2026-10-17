@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Project environment is missing. Run setup_windows.bat first.
  pause
  exit /b 1
)
echo Running Phase 1 audit...
.venv\Scripts\python.exe phase1_audit.py
if errorlevel 1 goto :error
echo.
echo Running Phase 2 split and EDA...
.venv\Scripts\python.exe phase2_split_eda.py
if errorlevel 1 goto :error
echo.
echo SUCCESS: Phase 2 finished. Read reports\phase2_eda_findings.md
pause
exit /b 0
:error
echo A script failed. Copy the full error message and share it with ChatGPT.
pause
exit /b 1
