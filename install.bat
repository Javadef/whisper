@echo off
echo ========================================
echo Whisper AI - Quick Start Installer
echo ========================================
echo.

REM Check Python
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] FFmpeg not found. Please install FFmpeg.
    echo Install with: choco install ffmpeg
    echo.
)

REM Create virtual environment if not exists
if not exist "venv\" (
    echo [1/5] Creating virtual environment...
    py -m venv venv
)

REM Activate virtual environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check CUDA
echo [3/5] Checking CUDA availability...
py -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo Installing PyTorch with CUDA 11.8...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
)

REM Install dependencies
echo [4/5] Installing dependencies...
pip install -r requirements.txt

REM Check for LLM model
echo [5/5] Checking for LLM model...
if not exist "models\*.gguf" (
    echo.
    echo [INFO] No LLM model found.
    echo Please download a model using: .\download_model.ps1
    echo Or manually download and place in models\ folder
    echo.
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download LLM model: .\download_model.ps1
echo 2. Run server: python run.py
echo 3. Open browser: http://localhost:8000
echo.
pause
