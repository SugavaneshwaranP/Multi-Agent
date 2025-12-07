@echo off
REM Launch Advanced Analytics Streamlit App
echo ========================================
echo Cognifyx Advanced Analytics Platform
echo ========================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"
E:\cognifyx\.venv\Scripts\python.exe -m streamlit run streamlit_app_advanced.py --server.port 8502

pause
