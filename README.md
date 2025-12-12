# Whisper AI - Comprehensive Transcription Platform

🚀 **AI-Powered Transcription with Live Streaming, LLM Chat & Multi-Language Translation**

Optimized for **NVIDIA RTX 4060** (8GB VRAM)

## ✨ Features

- 🎯 **Faster Whisper Turbo** - 8x faster transcription with high accuracy
- 🌍 **99+ Languages** - Auto-detect and transcribe any language
- 🤖 **LLM Integration** - Llama 3.2/Qwen for Q&A and advanced translation
- 📺 **Live Stream Support** - YouTube, Twitch, RTMP, HLS real-time transcription
- 🔄 **Multi-Language Translation** - Uzbek ↔ English ↔ Russian + more via LLM
- ⚡ **WebSocket Streaming** - Real-time continuous transcription
- 💬 **AI Chat Assistant** - Ask questions about transcriptions
- 🎨 **Modern Web UI** - Beautiful shadcn-inspired interface
- 📊 **Word Timestamps** - Precise word-level timing
- 🎥 **Multiple Formats** - MP4, MKV, MP3, WAV, FLAC, and more

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (RTX 4060 recommended)
- CUDA Toolkit 11.8+ and cuDNN
- FFmpeg installed and in PATH

## Installation

### 1. Install FFmpeg

```bash
# Windows (using chocolatey)
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

### 2. Create Virtual Environment

```bash
cd whisper
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

The repository also falls back to Ollama when a local GGUF model is not available — `app/services/llm_service.py` will use `app/services/ollama_service.py` automatically if needed.

## Running the Server

### Development Mode (with auto-reload)

```bash
python run_dev.py
```

### Production Mode

```bash
python run.py
```

Or with Uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

```bash
GET /health
```

### Transcribe (Sync)

```bash
POST /transcribe
Content-Type: multipart/form-data

file: <audio/video file>
task: transcribe | translate  # 'translate' uses Whisper's built-in EN translation
language: uz | en | ru | auto  # optional, auto-detect if not provided
translate_to: english,russian  # optional, additional translations
word_timestamps: true | false
```

### Transcribe (Async)

```bash
POST /transcribe/async
# Same parameters as /transcribe
# Returns: {"task_id": "uuid", "status": "pending"}
```

### Check Task Status

```bash
GET /task/{task_id}
# Returns task status and results when complete
```

### Translate Text

```bash
POST /translate
Content-Type: application/json

{
    "text": "Salom, bugun ob-havo juda yaxshi.",
    "source_language": "uzbek",
    "target_language": "english"
}
```

### Get Supported Languages

```bash
GET /languages
```

## Usage Examples

### Python

```python
import requests

# Transcribe a video
with open("video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={
            "task": "transcribe",
            "translate_to": "english,russian"
        }
    )
    
result = response.json()
print(result["transcription"])
print(result["translations"])
```

### cURL

```bash
# Transcribe with translation
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@video.mp4" \
  -F "task=transcribe" \
  -F "translate_to=english,russian"

# Translate text
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Salom", "source_language": "uzbek", "target_language": "english"}'
```

### Async Transcription

```python
import requests
import time

# Submit async task
with open("video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe/async",
        files={"file": f},
        data={"task": "transcribe"}
    )

task_id = response.json()["task_id"]

# Poll for results
while True:
    status = requests.get(f"http://localhost:8000/task/{task_id}").json()
    
    if status["status"] == "completed":
        print(status["result"]["transcription"])
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break
    
    time.sleep(2)
```

## Configuration

Edit `app/config.py` to customize:

```python
# Model size: tiny, base, small, medium, large-v2, large-v3
WHISPER_MODEL_SIZE = "large-v3"

# Compute type (for RTX 4060):
# - "float16": Best quality (default)
# - "int8_float16": Faster, less VRAM
# - "int8": Fastest, lowest VRAM
WHISPER_COMPUTE_TYPE = "float16"

# Max concurrent tasks
MAX_CONCURRENT_TASKS = 3

# Max file size (bytes)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
```

## Performance Tips

1. **Use VAD Filter**: Enabled by default, significantly speeds up processing
2. **Choose Right Model**: `large-v3` for best quality, `medium` for speed/quality balance
3. **Compute Type**: Use `int8_float16` if running low on VRAM
4. **Batch Processing**: Use async endpoint for multiple files

## API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

```bash
# Run test suite
python test_api.py

# Test with specific file
python test_api.py video.mp4 english,russian
```

## License

MIT
