@echo off
echo ============================================================
echo   Colony Counter - Installing dependencies
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.10+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo Installing dependencies...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Done! Now run the app: run.bat
echo ============================================================
echo.
pause
