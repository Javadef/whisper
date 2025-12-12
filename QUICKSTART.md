# 🚀 Quick Start Guide - Whisper AI

Get up and running in **5 minutes**!

## Prerequisites Check

✅ Python 3.10+ installed  
✅ NVIDIA GPU with CUDA support (RTX 4060 or better)  
✅ ~10GB free disk space  
✅ Internet connection for model downloads  

## Installation (Windows)

### Method 1: Automated (Recommended)

```powershell
# 1. Run installer
.\install.bat

# 2. Download LLM model
.\download_model.ps1

# 3. Start server
.\start.bat
```

**Done!** Open http://localhost:8000

### Method 2: Manual

```powershell
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# 2. Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model (choose one)
cd models
curl -L -o llama-3.2-3b-instruct.Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
cd ..

# 5. Run server
python run.py
```

## First Use

1. **Open Browser**: http://localhost:8000
2. **Check Status**: Sidebar shows GPU, Whisper, and LLM status
3. **Upload File**: Drag & drop video/audio file
4. **Try Chat**: Ask AI questions about your transcriptions
5. **Test Streaming**: Paste YouTube URL for live transcription

## Quick Test

```powershell
# Test with sample file
python test_api.py path\to\video.mp4 english,russian
```

## Features Overview

### 📤 Upload & Transcribe
- Drag & drop any video/audio file
- Auto-detect language (99+ languages)
- Get translations in English, Russian, Uzbek
- Export results

### 📺 Live Stream Transcription
- YouTube, Twitch, RTMP streams
- Real-time transcription
- Continuous or chunked processing
- Quality selection

### 💬 AI Chat Assistant
- Ask questions about transcriptions
- General Q&A capabilities
- Context-aware responses
- Conversation history

### 🌍 Translation
- Uzbek ↔ English ↔ Russian
- NLLB for fast translation
- LLM for better quality
- Multiple languages via LLM

## Common Commands

```powershell
# Start server
python run.py

# Development mode (auto-reload)
python run_dev.py

# Test API
python test_api.py

# Check GPU
nvidia-smi

# Check Python packages
pip list
```

## Configuration

Edit `app\config.py` for customization:

```python
# Whisper model (turbo = fast, large-v3 = accurate)
WHISPER_MODEL_SIZE = "turbo"

# LLM model path
LLM_MODEL_PATH = MODELS_DIR / "llama-3.2-3b-instruct.Q4_K_M.gguf"

# GPU settings
LLM_GPU_LAYERS = 33  # Higher = faster, more VRAM
WHISPER_COMPUTE_TYPE = "float16"  # or "int8_float16" for less VRAM

# Concurrent tasks
MAX_CONCURRENT_TASKS = 3
```

## Troubleshooting

### "CUDA not available"
```powershell
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Out of memory"
Edit config.py:
```python
WHISPER_MODEL_SIZE = "medium"
WHISPER_COMPUTE_TYPE = "int8_float16"
LLM_GPU_LAYERS = 20
```

### "FFmpeg not found"
```powershell
choco install ffmpeg
```

### "LLM not loading"
- Check model file exists in `models/` folder
- Verify filename matches config.py
- Try smaller model (Llama 3.2 3B)

## API Examples

### Python
```python
import requests

# Transcribe
with open("video.mp4", "rb") as f:
    r = requests.post("http://localhost:8000/transcribe", 
                      files={"file": f},
                      data={"translate_to": "english"})
    print(r.json()["transcription"])

# Chat
r = requests.post("http://localhost:8000/chat",
                  json={"message": "Hello!"})
print(r.json()["response"])
```

### cURL
```bash
# Transcribe
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@video.mp4" \
  -F "translate_to=english,russian"

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What languages do you support?"}'
```

### JavaScript
```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('translate_to', 'english');

fetch('http://localhost:8000/transcribe', {
    method: 'POST',
    body: formData
})
.then(r => r.json())
.then(data => console.log(data.transcription));

// WebSocket streaming
const ws = new WebSocket('ws://localhost:8000/ws/stream');
ws.send(JSON.stringify({
    action: 'start',
    stream_url: 'https://youtube.com/watch?v=...'
}));
ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    console.log(data.text);
};
```

## Performance Tips

### For Speed:
- Use Whisper `turbo` model
- Use Llama 3.2 3B LLM
- Set compute type to `int8_float16`

### For Quality:
- Use Whisper `large-v3` model
- Use Qwen 2.5 7B or Gemma 2 9B LLM
- Set compute type to `float16`

### For Multiple Files:
- Use async endpoint: `/transcribe/async`
- Increase `MAX_CONCURRENT_TASKS`
- Monitor GPU memory with `nvidia-smi`

## Supported Formats

**Audio:** MP3, WAV, FLAC, OGG, M4A, AAC  
**Video:** MP4, MKV, AVI, MOV, WebM, FLV  
**Streams:** YouTube, Twitch, RTMP, HLS, HTTP  

## Documentation

- **Full README**: [README.md](README.md)
- **Setup Guide**: [SETUP.md](SETUP.md)
- **Project Structure**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **API Docs**: http://localhost:8000/docs
- **Models Guide**: [models/README.md](models/README.md)

## Getting Help

1. Check terminal output for errors
2. Review [SETUP.md](SETUP.md) for detailed instructions
3. Verify GPU with `nvidia-smi`
4. Check model files in `models/` folder
5. Test individual components with `test_api.py`

## What's Next?

✨ **Explore Features**
- Upload different file types
- Try live stream transcription
- Chat with AI assistant
- Test multiple languages

🎯 **Customize**
- Change models in config.py
- Modify UI in static/ folder
- Add new endpoints to main.py

📊 **Monitor**
- Watch GPU usage: `nvidia-smi -l 1`
- Check processing times in logs
- Review results quality

---

**Enjoy transcribing with Whisper AI! 🎙️🚀**

Questions? Check the docs or API documentation at `/docs`
