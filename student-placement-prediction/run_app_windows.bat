@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 echo The app stopped with an error. Copy the error message and send it to ChatGPT.
pause
