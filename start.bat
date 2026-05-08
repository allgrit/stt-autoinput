@echo off
cd /d "%~dp0"
"C:\Program Files\Git\bin\bash.exe" -c "cd '/c/Users/allgrit/Documents/codex/STL-autoinput' && '/c/Users/allgrit/AppData/Local/Programs/Python/Python311/python.exe' -u -c \"exec(open('dictate_realtime.py', encoding='utf-8').read())\""
if errorlevel 1 pause
