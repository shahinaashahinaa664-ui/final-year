@echo off
cd /d "%~dp0\\react-dashboard"

if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install
)

echo Starting React dashboard on http://localhost:5173
npm run dev

pause
