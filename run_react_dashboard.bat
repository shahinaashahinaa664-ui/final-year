@echo off
cd /d "%~dp0"

echo Starting backend and frontend in separate windows...
start "Backend API" cmd /k call "%~dp0run_backend.bat"
start "React Dashboard" cmd /k call "%~dp0run_frontend.bat"

echo.
echo Open this URL in browser after both windows start:
echo http://localhost:5173
pause
