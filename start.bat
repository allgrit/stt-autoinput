@echo off
cd /d "%~dp0"
call "%~dp0find_python.bat"
echo [%date% %time%] Starting STT Autoinput > "%~dp0stl_log.txt"
echo Python: "%PYTHON%" >> "%~dp0stl_log.txt"
"%PYTHON%" -u "%~dp0dictate_realtime.py" >> "%~dp0stl_log.txt" 2>&1
echo [%date% %time%] Exited with code %errorlevel% >> "%~dp0stl_log.txt"
