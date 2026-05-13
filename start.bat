@echo off
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
echo [%date% %time%] Starting STT Autoinput > "%~dp0stl_log.txt"
echo Python: "%PYTHON%" >> "%~dp0stl_log.txt"
"%PYTHON%" -u "%~dp0dictate_realtime.py" >> "%~dp0stl_log.txt" 2>&1
echo [%date% %time%] Exited with code %errorlevel% >> "%~dp0stl_log.txt"
