@echo off
echo 🚀 Starting NASA Server with Email Integration
echo ======================================================

cd /d "C:\Users\91995\OneDrive\Desktop\NASA\Pipeline"

echo Starting server in background...
start "NASA Server" cmd /c "python nasa_server.py"

echo Waiting for server to start...
timeout /t 5 /nobreak >nul

echo Testing email integration...
python test_email_integration.py

echo.
echo ✅ Test complete! 
echo 🌐 Server should be running at: http://localhost:8080
echo 📧 Email setup guide: http://localhost:8080/email/setup
echo.
pause