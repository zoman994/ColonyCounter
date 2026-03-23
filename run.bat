@echo off
cd /d "%~dp0"
python colony_counter_v1.1.py
if errorlevel 1 (
    echo.
    echo ERROR: App crashed. Make sure you ran setup.bat first.
    pause
)
