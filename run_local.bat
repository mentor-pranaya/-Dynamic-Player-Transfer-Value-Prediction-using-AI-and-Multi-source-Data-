@echo off
REM Start backend and frontend for TransferIQ prototype
REM Open two separate PowerShell windows if you want to see logs, or run this file which opens new windows.

start powershell -NoExit -Command "cd '%~dp0' ; uvicorn backend:app --reload"
start powershell -NoExit -Command "cd '%~dp0' ; streamlit run app.py --server.port 8501"
echo Launched backend and frontend in two new PowerShell windows.
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:8501
pause
