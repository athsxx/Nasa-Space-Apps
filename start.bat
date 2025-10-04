@echo off
title Air Quality Monitoring Agent

echo.
echo ========================================
echo   Air Quality Monitoring Agent
echo ========================================
echo.

:menu
echo Choose an option:
echo.
echo [1] Install/Setup (First time only)
echo [2] Start Web API Server
echo [3] Run CLI (Interactive)
echo [4] Run Tests
echo [5] Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto server
if "%choice%"=="3" goto cli
if "%choice%"=="4" goto tests
if "%choice%"=="5" goto exit
goto menu

:install
echo.
echo Installing dependencies...
python install.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Installation failed! Make sure Python is installed and in PATH.
    pause
    goto menu
)
echo.
echo Installation completed!
pause
goto menu

:server
echo.
echo Starting Web API Server...
echo Visit http://localhost:8000 in your browser
echo Press Ctrl+C to stop the server
echo.
python main.py
pause
goto menu

:cli
echo.
echo Air Quality CLI - Interactive Mode
echo.
set /p location="Enter location (or press Enter for auto-detect): "
if "%location%"=="" (
    python cli.py --auto
) else (
    python cli.py --location "%location%"
)
echo.
pause
goto menu

:tests
echo.
echo Running tests...
python tests/test_aqi.py
echo.
pause
goto menu

:exit
echo.
echo Goodbye!
exit