@echo off
cd /d "%~dp0"
call "%~dp0find_python.bat"
"%PYTHON%" "%~dp0configure.py"
