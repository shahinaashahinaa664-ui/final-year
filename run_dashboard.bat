@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found at .venv\Scripts\python.exe
    echo Create it first, then install requirements.
    pause
    exit /b 1
)

echo Starting dashboard...
".venv\Scripts\python.exe" -m streamlit run dashboard.py

pause
