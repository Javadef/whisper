@echo off
echo Starting Whisper AI Server...
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Virtual environment not found. Running installer...
    call install.bat
)

REM Activate venv
call venv\Scripts\activate.bat

REM Start server
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop
echo.

python run.py
