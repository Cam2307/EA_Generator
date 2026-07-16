@echo off
rem Launch the MQL5 EA Factory & Curation Dashboard (unified discovery UI).
rem Browser refresh alone does NOT reload Python modules — this script
rem restarts Streamlit so discovery_panel.py changes always take effect.
cd /d "%~dp0"

echo Stopping any existing dashboard on port 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

if exist ".venv\Scripts\python.exe" (
    echo Starting dashboard via .venv — discovery UI 2026-07-10-duration
    ".venv\Scripts\python.exe" -m streamlit run app\dashboard.py
) else (
    echo Starting dashboard via PATH streamlit — discovery UI 2026-07-10-duration
    streamlit run app\dashboard.py
)
