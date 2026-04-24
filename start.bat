@echo off
cd /d "%~dp0"
where py >nul 2>&1 && (
    py -3 -u -c "exec(open('dictate_realtime.py', encoding='utf-8').read())"
) || (
    python -u -c "exec(open('dictate_realtime.py', encoding='utf-8').read())"
)
if errorlevel 1 pause
