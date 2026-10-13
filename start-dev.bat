@echo off
title MediRAG Launcher
REM NOTE: keep this file ASCII-only. cmd.exe mis-parses UTF-8 Chinese in .bat files.

echo ============================================
echo   MediRAG Dev Launcher
echo ============================================
echo.

set "ROOT=%~dp0"

if not exist "%ROOT%backend\.venv\Scripts\python.exe" (
    echo [ERROR] backend\.venv not found. Run start-init.bat first.
    pause
    exit /b 1
)
if not exist "%ROOT%backend\.env" (
    echo [ERROR] backend\.env not found. Copy .env.example and set API keys.
    pause
    exit /b 1
)

echo [1/2] Starting backend FastAPI (new window, port 8000, auto-reload on)...
start "MediRAG-Backend" cmd /k "cd /d %ROOT%backend && .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

REM wait for backend before frontend (frontend proxies /api to 8000)
timeout /t 5 /nobreak >nul

echo [2/2] Starting frontend Vite (new window, port 5173)...
start "MediRAG-Frontend" cmd /k "cd /d %ROOT%frontend && npm run dev"

echo.
echo Backend : http://127.0.0.1:8000   (API docs at /docs)
echo Frontend: http://localhost:5173   (open this in browser)
echo.
echo Accounts: admin/admin123 (admin)   doctor/doctor123 (doctor)
echo Close each window to stop the service.
echo.
pause
