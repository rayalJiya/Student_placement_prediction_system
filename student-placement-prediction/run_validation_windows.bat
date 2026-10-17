@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" validate_project.py
if errorlevel 1 (
  echo Validation found an issue. Copy the report or error and send it to ChatGPT.
) else (
  echo Validation passed. See reports\final_project_validation.json
)
pause
