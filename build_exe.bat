@echo off
echo ============================================================
echo   Colony Counter - Build standalone .exe
echo ============================================================
echo.
echo Installing PyInstaller...
python -m pip install pyinstaller --quiet

echo.
echo Building executable...
pyinstaller --onefile --windowed --name "ColonyCounter" colony_counter.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Done! File: dist\ColonyCounter.exe
echo   This .exe runs on any Windows PC without Python.
echo ============================================================
pause
