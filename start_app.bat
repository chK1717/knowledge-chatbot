@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo   Lancement SDV Chatbot Streamlit

echo ==========================================

where py >nul 2>&1
if errorlevel 1 (
  echo [ERREUR] Python Launcher py introuvable.
  echo Installe Python 3.11 puis relance ce script.
  pause
  exit /b 1
)

py -3.11 -m streamlit run app_streamlit.py

if errorlevel 1 (
  echo.
  echo [ERREUR] Le lancement de Streamlit a echoue.
  pause
)

endlocal
