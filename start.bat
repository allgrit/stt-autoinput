@echo off
cd /d "%~dp0"
rem Старый экземпляр держит stl_log.txt открытым: пока он жив, перенаправление
rem вывода в лог молча не срабатывает и новый Python не запускается.
rem Поэтому убиваем его здесь, до первой записи в лог.
if exist "%~dp0stl.pid" (
    for /f "usebackq delims=" %%p in ("%~dp0stl.pid") do taskkill /PID %%p /F >nul 2>&1
    del "%~dp0stl.pid" >nul 2>&1
    timeout /t 1 /nobreak >nul
)
call "%~dp0find_python.bat"
echo [%date% %time%] Starting STT Autoinput > "%~dp0stl_log.txt"
echo Python: "%PYTHON%" >> "%~dp0stl_log.txt"
"%PYTHON%" -u "%~dp0dictate_realtime.py" >> "%~dp0stl_log.txt" 2>&1
echo [%date% %time%] Exited with code %errorlevel% >> "%~dp0stl_log.txt"
