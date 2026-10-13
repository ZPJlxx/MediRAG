@echo off
title MediRAG Init
REM NOTE: keep this file ASCII-only. cmd.exe mis-parses UTF-8 Chinese in .bat files.

echo ============================================
echo   MediRAG One-time Init
echo ============================================
echo.
set "ROOT=%~dp0"

echo [1/3] Setting up backend environment...
cd /d "%ROOT%backend"

if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] python venv creation failed. Install Python 3.11+ first.
        pause
        exit /b 1
    )
)

call .venv\Scripts\python.exe -m pip install --quiet --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check network and retry.
    pause
    exit /b 1
)

if not exist ".env" (
    copy .env.example .env >nul
    echo.
    echo   *** IMPORTANT: edit backend\.env and set DASHSCOPE_API_KEY ***
    echo.
)

echo [2/3] Creating tables + seed accounts + demo knowledge base...
call .venv\Scripts\python.exe -m app.seed
if errorlevel 1 (
    echo [ERROR] app.seed failed.
    pause
    exit /b 1
)

echo [3/3] Installing frontend dependencies...
cd /d "%ROOT%frontend"
call npm install
if errorlevel 1 (
    echo [TIP] npm install failed. Try: npm install --cache %%TEMP%%
pmcache
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Init done! Double-click start-dev.bat
echo ============================================
pause
