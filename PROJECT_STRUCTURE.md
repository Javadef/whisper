# Whisper AI - Project Structure

```
whisper/
│
├── app/                              # Main application package
│   ├── __init__.py
│   ├── config.py                     # Configuration settings
│   ├── models.py                     # Pydantic request/response models
│   ├── main.py                       # FastAPI application & endpoints
│   │
│   └── services/                     # Business logic services
│       ├── __init__.py
│       ├── whisper_service.py        # Faster Whisper transcription
│       ├── translation_service.py    # NLLB translation service
│       ├── llm_service.py           # LLM for chat & advanced translation
│       ├── stream_service.py        # Live stream processing
│       └── task_manager.py          # Async task management
│
├── static/                           # Web interface files
│   ├── index.html                    # Main HTML page
│   ├── style.css                     # Modern shadcn-inspired CSS
│   └── app.js                        # Frontend JavaScript
│
├── models/                           # AI models storage
│   ├── README.md                     # Model download instructions
│   └── .gitkeep
│   └── (place .gguf models here)
│
├── uploads/                          # Temporary uploaded files
│   └── (auto-created)
│
├── outputs/                          # Processed outputs
│   └── (auto-created)
│
├── stream_chunks/                    # Temporary stream chunks
│   └── (auto-created)
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Main documentation
├── SETUP.md                          # Detailed setup guide
│
├── run.py                           # Production server launcher
├── run_dev.py                       # Development server (auto-reload)
├── test_api.py                      # API test suite
│
├── install.bat                      # Windows installer script
├── start.bat                        # Quick start script
└── download_model.ps1              # PowerShell model downloader
```

## File Descriptions

### Core Application

- **app/config.py** - All configuration settings (models, paths, GPU settings)
- **app/models.py** - Pydantic schemas for API requests/responses
- **app/main.py** - FastAPI app with all endpoints (transcription, chat, streaming)

### Services (Business Logic)

- **whisper_service.py** - Handles audio/video transcription using Faster Whisper
- **translation_service.py** - NLLB-based translation for Uzbek/English/Russian
- **llm_service.py** - LLM integration for chat, Q&A, and advanced translation
- **stream_service.py** - Live stream processing (YouTube, Twitch, RTMP, etc.)
- **task_manager.py** - Manages async background tasks with progress tracking

### Web Interface

- **static/index.html** - Modern single-page application
- **static/style.css** - Dark theme, shadcn-inspired design
- **static/app.js** - Client-side logic (upload, streaming, chat, WebSockets)

### Utilities

- **run.py** - Start production server (single worker, GPU optimized)
- **run_dev.py** - Development server with auto-reload
- **test_api.py** - Test all API endpoints
- **install.bat** - Automated setup for Windows
- **start.bat** - One-click server start
- **download_model.ps1** - Interactive model downloader

## Key Features by File

### Transcription Pipeline
```
User Upload → app/main.py (/transcribe)
           → whisper_service.py (transcribe)
           → translation_service.py (translate)
           → Response with results
```

### Live Streaming
```
Stream URL → app/main.py (/ws/stream)
          → stream_service.py (download chunks)
          → whisper_service.py (transcribe chunks)
          → WebSocket → Client (real-time results)
```

### AI Chat
```
User Message → app/main.py (/chat)
            → llm_service.py (generate response)
            → Response with AI answer
```

## Configuration Hierarchy

1. **Default Settings** - In app/config.py
2. **Environment Variables** - Override via os.getenv()
3. **Runtime Changes** - Update config.py and restart

## Data Flow

### File Upload Transcription
1. Client uploads file via web interface
2. File saved to `uploads/` directory
3. Whisper processes file (GPU accelerated)
4. Optional: Translations via NLLB or LLM
5. Results returned to client
6. File deleted from uploads

### Stream Processing
1. Client provides stream URL
2. Streamlink extracts stream
3. FFmpeg downloads chunks to `stream_chunks/`
4. Each chunk transcribed in real-time
5. Results sent via WebSocket
6. Old chunks automatically cleaned up

### Chat with Context
1. User sends message + optional context (transcription)
2. LLM processes with conversation history
3. Response generated (GPU accelerated)
4. History maintained per session

## API Endpoint Map

### Transcription
- `POST /transcribe` - Sync file transcription
- `POST /transcribe/async` - Async file transcription
- `GET /task/{id}` - Check async task status

### Streaming
- `POST /stream/transcribe` - Single chunk from stream
- `GET /stream/info` - Get stream information
- `WS /ws/stream` - Real-time streaming WebSocket

### AI & Translation
- `POST /chat` - Chat with AI assistant
- `POST /chat/clear` - Clear conversation history
- `POST /summarize` - Summarize transcription
- `POST /translate` - NLLB translation
- `POST /translate/llm` - LLM-based translation

### System
- `GET /` - Web interface
- `GET /health` - System status
- `GET /languages` - Supported languages
- `GET /docs` - API documentation

## Technology Stack

### Backend
- **FastAPI** - Async web framework
- **Uvicorn** - ASGI server
- **Faster Whisper** - Transcription engine
- **llama.cpp** - LLM inference
- **Streamlink** - Stream extraction
- **FFmpeg** - Media processing

### Frontend
- **Vanilla JavaScript** - No framework overhead
- **WebSocket API** - Real-time communication
- **CSS Grid/Flexbox** - Modern layouts
- **Custom CSS** - shadcn-inspired design

### AI Models
- **Whisper Turbo** - 809MB, 8x faster
- **Llama 3.2 3B** - Fast LLM (2GB)
- **Qwen 2.5 7B** - Better multilingual (4.4GB)
- **NLLB-200** - Translation (600M params)

## Customization Points

### Add New Endpoint
1. Define Pydantic model in `app/models.py`
2. Add route in `app/main.py`
3. Implement logic in service file
4. Update frontend in `static/app.js`

### Add New Model
1. Download GGUF model to `models/`
2. Update `LLM_MODEL_PATH` in config.py
3. Adjust `LLM_GPU_LAYERS` for VRAM
4. Restart server

### Modify UI
1. Edit HTML structure in `static/index.html`
2. Update styles in `static/style.css`
3. Add functionality in `static/app.js`
4. Refresh browser (Ctrl+F5)

## Performance Monitoring

### Check GPU Usage
```powershell
nvidia-smi -l 1  # Update every second
```

### Check Memory
```python
# Add to endpoints for debugging
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

### Profile Endpoints
```python
import time
start = time.time()
# ... code ...
print(f"Took: {time.time()-start:.2f}s")
```

## Security Considerations

1. **File Upload Validation** - Size limits, format checks
2. **Rate Limiting** - Max concurrent tasks
3. **Input Sanitization** - Prevent injection attacks
4. **CORS Configuration** - Adjust for production
5. **API Authentication** - Add if needed (not included)

## Deployment Notes

### Local/Development
- Use `run_dev.py` for auto-reload
- Access via localhost:8000

### Production
- Use `run.py` with single worker (GPU limitation)
- Configure firewall for port 8000
- Consider NGINX reverse proxy
- Add HTTPS with certificates
- Implement authentication

---

**Project successfully transformed into a comprehensive AI platform! 🚀**
