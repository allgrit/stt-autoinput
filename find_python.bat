@echo off
rem Ищет интерпретатор, в котором реально стоят зависимости проекта.
rem Результат возвращается в переменной PYTHON вызывающего скрипта.
rem Порядок: .venv проекта -> Python 3.11 в LOCALAPPDATA -> py -3.11 -> python из PATH.
rem Кандидат принимается только если в нём импортируется sounddevice:
rem голый PATH может указывать на другую версию Python без зависимостей.

set "PYTHON="
call :try "%~dp0.venv\Scripts\python.exe"
if not defined PYTHON call :try "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON for /f "delims=" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do call :try "%%i"
if not defined PYTHON call :try "python"

rem Ни один кандидат не подошёл - берём python из PATH, чтобы ошибка попала в лог.
if not defined PYTHON set "PYTHON=python"
goto :eof

:try
if "%~1"=="" goto :eof
if /i not "%~1"=="python" if not exist "%~1" goto :eof
"%~1" -c "import sounddevice" >nul 2>&1
if errorlevel 1 goto :eof
set "PYTHON=%~1"
goto :eof
