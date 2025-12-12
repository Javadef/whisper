# Whisper AI - Setup Guide

## Complete Installation Steps

### Step 1: Install Prerequisites

1. **Install Python 3.10+**
   - Download from https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

2. **Install FFmpeg**
   ```powershell
   # Using Chocolatey (recommended)
   choco install ffmpeg
   
   # Or download manually from https://ffmpeg.org/download.html
   # Add to PATH: C:\ffmpeg\bin
   ```

3. **Install CUDA Toolkit** (if not already installed)
   - Check CUDA version: `nvidia-smi`
   - Download from https://developer.nvidia.com/cuda-downloads
   - Recommended: CUDA 11.8 or 12.1

### Step 2: Create Virtual Environment

```powershell
cd c:\Users\Java\Desktop\whisper
python -m venv venv
.\venv\Scripts\Activate
```

### Step 3: Install PyTorch with CUDA

```powershell
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 4: Install Project Dependencies

```powershell
pip install -r requirements.txt
```

**Note:** Installing `llama-cpp-python` with CUDA support:
```powershell
# If the above doesn't work, try:
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

### Step 5: Download LLM Model

**Option A: Using PowerShell Script (Recommended)**
```powershell
.\download_model.ps1
```

**Option B: Manual Download**
```powershell
# Navigate to models folder
cd models

# Download Llama 3.2 3B (Recommended)
curl -L -o llama-3.2-3b-instruct.Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

cd ..
```

### Step 6: Update Configuration

Edit `app\config.py` and set your model:

```python
# Line 24 - Set your downloaded model
LLM_MODEL_PATH = MODELS_DIR / "llama-3.2-3b-instruct.Q4_K_M.gguf"
```

### Step 7: Run the Application

```powershell
# Development mode (with auto-reload)
python run_dev.py

# Production mode
python run.py
```

Open browser: **http://localhost:8000**

## Verification Checklist

✅ **Check Python:**
```powershell
python --version  # Should be 3.10+
```

✅ **Check CUDA:**
```powershell
nvidia-smi  # Should show GPU info
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

✅ **Check FFmpeg:**
```powershell
ffmpeg -version  # Should show version
```

✅ **Check Dependencies:**
```powershell
pip list | findstr "torch fastapi faster-whisper llama-cpp-python"
```

✅ **Check LLM Model:**
```powershell
dir models\*.gguf  # Should show your model file
```

✅ **Test API:**
```powershell
# In another terminal after starting server
curl http://localhost:8000/health
```

## Common Issues & Solutions

### Issue: "CUDA not available"

**Solution:**
```powershell
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "llama-cpp-python not compiled with CUDA"

**Solution:**
```powershell
pip uninstall llama-cpp-python
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --no-cache-dir
```

### Issue: "FFmpeg not found"

**Solution:**
```powershell
# Add to PATH or install via Chocolatey
choco install ffmpeg

# Or download and extract to C:\ffmpeg
# Add C:\ffmpeg\bin to System PATH
```

### Issue: "Out of memory"

**Solution - Edit `app\config.py`:**
```python
# Reduce GPU usage
WHISPER_MODEL_SIZE = "medium"  # Instead of turbo
WHISPER_COMPUTE_TYPE = "int8_float16"  # Instead of float16
LLM_GPU_LAYERS = 20  # Instead of 33

# Or use smaller LLM model (Llama 3.2 3B instead of Qwen 7B)
```

### Issue: "Module not found"

**Solution:**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Port 8000 already in use"

**Solution:**
```powershell
# Change port in run.py or run_dev.py
# Or kill existing process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## Performance Optimization

### For RTX 4060 (8GB VRAM):

1. **Best Speed:**
   ```python
   WHISPER_MODEL_SIZE = "turbo"
   WHISPER_COMPUTE_TYPE = "int8_float16"
   LLM_MODEL_PATH = MODELS_DIR / "llama-3.2-3b-instruct.Q4_K_M.gguf"
   LLM_GPU_LAYERS = 33
   ```

2. **Best Quality:**
   ```python
   WHISPER_MODEL_SIZE = "large-v3"
   WHISPER_COMPUTE_TYPE = "float16"
   LLM_MODEL_PATH = MODELS_DIR / "qwen2.5-7b-instruct.Q4_K_M.gguf"
   LLM_GPU_LAYERS = 33
   ```

3. **Balanced:**
   ```python
   WHISPER_MODEL_SIZE = "turbo"
   WHISPER_COMPUTE_TYPE = "float16"
   LLM_MODEL_PATH = MODELS_DIR / "llama-3.2-3b-instruct.Q4_K_M.gguf"
   LLM_GPU_LAYERS = 33
   ```

## Testing

```powershell
# Run test suite
python test_api.py

# Test with a file
python test_api.py path\to\video.mp4 english,russian
```

## Next Steps

1. ✅ Open web interface: http://localhost:8000
2. ✅ Check system status in sidebar
3. ✅ Upload a test file
4. ✅ Try live stream transcription
5. ✅ Chat with AI assistant
6. ✅ Check API docs: http://localhost:8000/docs

## Getting Help

- Check console output for error messages
- Review logs in terminal
- Verify all dependencies are installed
- Ensure GPU drivers are up to date
- Check README.md for more details

---

**Ready to start transcribing! 🚀**
