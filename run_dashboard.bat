@echo off
rem Launch the MQL5 EA Factory & Curation Dashboard
cd /d "%~dp0"
if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app\dashboard.py
) else (
    streamlit run app\dashboard.py
)
