@echo off
cd /d "%~dp0"
start "" cmd /c "timeout /t 5 /nobreak >nul & start chrome http://localhost:8501"
py -m streamlit run app.py --server.headless true
pause