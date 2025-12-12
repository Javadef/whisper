"""Main FastAPI application for Whisper transcription and translation."""
import os
import time
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
import json

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import aiofiles

from app.config import (
    UPLOAD_DIR,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE,
    MAX_CONCURRENT_TASKS,
    WHISPER_MODEL_SIZE,
    OLLAMA_MODEL,
)
import shutil
import asyncio as _asyncio
import asyncio
import subprocess
import traceback
from app.models import (
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
    TaskStatus,
    HealthResponse,
    TargetLanguage,
    Segment,
    ChatRequest,
    ChatResponse,
    StreamTranscribeRequest,
    SummarizeRequest,
    LLMTranslateRequest,
)
from app.services.whisper_service import whisper_service, WhisperService
from app.services.translation_service import translation_service, TranslationService
from app.services.task_manager import task_manager
from app.services.llm_service import llm_service, LLMService
from app.services.stream_service import stream_service

# Progress tracking store
progress_store = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load models on startup."""
    logger.info("Starting up Whisper API...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Pre-load Whisper model
    logger.info("Pre-loading Whisper model...")
    WhisperService.get_model()
    
    # Try to load LLM model
    logger.info("Loading LLM model...")
    LLMService.get_model()
    
    logger.info("Startup complete!")
    yield
    
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Whisper AI - Transcription, Translation & Q&A",
    description="AI-powered transcription with live streaming, LLM chat, and multi-language translation",
    version="2.0.0",
    lifespan=lifespan,
)

# Mount static files for web interface
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and GPU status."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    
    # Get LLM model name
    from app.config import LLM_MODEL_PATH
    llm_model_name = LLM_MODEL_PATH.stem if LLM_MODEL_PATH.exists() else None
    
    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "models_loaded": WhisperService.is_loaded(),
        "llm_loaded": LLMService.is_loaded(),
        "whisper_model": WHISPER_MODEL_SIZE,
        "llm_model": llm_model_name,
    }


@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Stream progress updates via SSE."""
    async def event_generator():
        try:
            while True:
                if task_id in progress_store:
                    progress_data = progress_store[task_id]
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    
                    if progress_data.get('status') in ['completed', 'error']:
                        break
                
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    index_file = Path(__file__).parent.parent / "static" / "index.html"
    
    if index_file.exists():
        async with aiofiles.open(index_file, "r", encoding="utf-8") as f:
            return await f.read()
    
    return """
    <html><head><title>Whisper AI</title></head><body>
    <h1>Whisper AI API</h1>
    <p>Visit <a href="/docs">/docs</a> for API documentation</p>
    <p>Web interface not found. Create static/index.html</p>
    </body></html>
    """


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    task: str = Form("transcribe", description="Task: 'transcribe' or 'translate' (to English)"),
    language: Optional[str] = Form(None, description="Source language code (auto-detect if not provided)"),
    translate_to: Optional[str] = Form(None, description="Comma-separated target languages: 'english,russian'"),
    word_timestamps: bool = Form(False, description="Include word-level timestamps"),
):
    """
    Transcribe an audio or video file.
    
    - **file**: Audio/video file (mp4, mkv, mp3, wav, etc.)
    - **task**: 'transcribe' for transcription, 'translate' for Whisper's built-in translation to English
    - **language**: Source language code (e.g., 'uz', 'en', 'ru'). Auto-detected if not provided.
    - **translate_to**: Additional translation targets: 'english', 'russian' (comma-separated)
    - **word_timestamps**: Include word-level timing information
    """
    start_time = time.time()
    task_id = str(uuid.uuid4())
    
    # Initialize progress
    progress_store[task_id] = {
        'status': 'uploading',
        'progress': 0,
        'message': 'Uploading file...',
        'task_id': task_id
    }
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {SUPPORTED_FORMATS}",
        )
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{int(time.time())}_{file.filename}"
    
    try:
        # Check file size while saving
        total_size = 0
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB",
                    )
                await f.write(chunk)
        
        logger.info(f"Processing file: {file.filename} ({total_size / 1024 / 1024:.1f}MB)")
        
        # Update progress
        progress_store[task_id] = {
            'status': 'processing',
            'progress': 10,
            'message': 'Loading audio...',
            'task_id': task_id
        }
        
        # Acquire semaphore to limit concurrent processing
        semaphore = task_manager.get_semaphore(MAX_CONCURRENT_TASKS)
        
        async with semaphore:
            # Update progress
            progress_store[task_id] = {
                'status': 'processing',
                'progress': 30,
                'message': 'Transcribing audio...',
                'task_id': task_id
            }
            
            # Transcribe
            segments, metadata = await whisper_service.transcribe(
                str(file_path),
                task=task,
                language=language,
                word_timestamps=word_timestamps,
            )
            
            # Update progress
            progress_store[task_id] = {
                'status': 'processing',
                'progress': 70,
                'message': 'Processing results...',
                'task_id': task_id
            }
        
        # Combine segments into full transcription
        full_transcription = " ".join(seg.text for seg in segments)
        
        # Update progress
        progress_store[task_id] = {
            'status': 'processing',
            'progress': 80,
            'message': 'Preparing translations...',
            'task_id': task_id
        }
        
        # Handle additional translations
        translations = None
        if translate_to:
            target_langs = [lang.strip().lower() for lang in translate_to.split(",")]
            detected_lang = metadata["language"]
            
            # Map detected language to our format
            source_lang_map = {
                "uz": "uzbek",
                "en": "english", 
                "ru": "russian",
            }
            source_lang = source_lang_map.get(detected_lang, detected_lang)
            
            translations = await translation_service.translate_to_multiple(
                full_transcription,
                source_lang,
                target_langs,
            )
        
        processing_time = time.time() - start_time
        
        response = TranscriptionResponse(
            success=True,
            filename=file.filename,
            detected_language=metadata["language"],
            language_probability=metadata["language_probability"],
            duration=metadata["duration"],
            transcription=full_transcription,
            segments=segments,
            translations=translations,
            processing_time=processing_time,
        )
        
        logger.info(f"Completed: {file.filename} in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if file_path.exists():
            file_path.unlink()


@app.post("/transcribe/async")
async def transcribe_file_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    translate_to: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
):
    """
    Start async transcription and return task ID immediately.
    Use /task/{task_id} to check status and get results.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}",
        )
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{int(time.time())}_{file.filename}"
    
    try:
        total_size = 0
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail="File too large")
                await f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Create task
    task_id = task_manager.create_task()
    
    # Schedule background processing
    background_tasks.add_task(
        process_transcription_task,
        task_id,
        str(file_path),
        file.filename,
        task,
        language,
        translate_to,
        word_timestamps,
    )
    
    return {"task_id": task_id, "status": "pending"}


async def process_transcription_task(
    task_id: str,
    file_path: str,
    filename: str,
    task: str,
    language: Optional[str],
    translate_to: Optional[str],
    word_timestamps: bool,
):
    """Background task for processing transcription."""
    start_time = time.time()
    
    try:
        task_manager.update_task(task_id, status="processing", progress=0.1)
        
        semaphore = task_manager.get_semaphore(MAX_CONCURRENT_TASKS)
        
        async with semaphore:
            task_manager.update_task(task_id, progress=0.2)
            
            segments, metadata = await whisper_service.transcribe(
                file_path,
                task=task,
                language=language,
                word_timestamps=word_timestamps,
            )
        
        task_manager.update_task(task_id, progress=0.8)
        
        full_transcription = " ".join(seg.text for seg in segments)
        
        translations = None
        if translate_to:
            target_langs = [lang.strip().lower() for lang in translate_to.split(",")]
            source_lang_map = {"uz": "uzbek", "en": "english", "ru": "russian"}
            source_lang = source_lang_map.get(metadata["language"], metadata["language"])
            
            translations = await translation_service.translate_to_multiple(
                full_transcription,
                source_lang,
                target_langs,
            )
        
        processing_time = time.time() - start_time
        
        result = TranscriptionResponse(
            success=True,
            filename=filename,
            detected_language=metadata["language"],
            language_probability=metadata["language_probability"],
            duration=metadata["duration"],
            transcription=full_transcription,
            segments=segments,
            translations=translations,
            processing_time=processing_time,
        )
        
        task_manager.update_task(task_id, status="completed", progress=1.0, result=result)
        logger.info(f"Task {task_id} completed in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        task_manager.update_task(task_id, status="failed", error=str(e))
    finally:
        # Cleanup
        path = Path(file_path)
        if path.exists():
            path.unlink()


@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status and result of an async transcription task."""
    status = task_manager.get_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text between languages.
    
    Supported languages: uzbek (uz), english (en), russian (ru)
    """
    try:
        translated = await translation_service.translate(
            request.text,
            request.source_language,
            request.target_language,
        )
        
        return TranslationResponse(
            success=True,
            source_text=request.text,
            translated_text=translated,
            source_language=request.source_language,
            target_language=request.target_language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages for translation."""
    return {
        "transcription_languages": "Auto-detected (supports 99+ languages)",
        "translation_languages": {
            "uzbek": "uz",
            "english": "en", 
            "russian": "ru",
        },
        "supported_formats": list(SUPPORTED_FORMATS),
    }


# ===== LLM & Chat Endpoints =====

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the AI assistant.
    
    - **message**: Your message/question
    - **session_id**: Session ID for conversation history
    - **context**: Optional context (e.g., transcription text)
    """
    try:
        if request.context:
            # Answer based on context
            response = await llm_service.answer_question(request.message, request.context)
        else:
            # General chat
            response = await llm_service.chat(request.message, request.session_id)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/clear")
async def clear_chat_history(session_id: str = "default"):
    """Clear conversation history for a session."""
    llm_service.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """Summarize text using LLM."""
    try:
        summary = await llm_service.summarize_transcription(
            request.text,
            request.language,
        )
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/llm")
async def translate_with_llm(request: LLMTranslateRequest):
    """
    Translate using LLM (better quality, supports more languages).
    """
    try:
        translation = await llm_service.translate_with_llm(
            request.text,
            request.source_language,
            request.target_language,
        )
        
        return {
            "success": True,
            "source_text": request.text,
            "translated_text": translation,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "method": "llm",
        }
    except Exception as e:
        logger.error(f"LLM translation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ollama/status")
async def ollama_status():
    """Return local Ollama CLI status (models and running processes).

    Useful for frontend to display whether Ollama is available and which model is running.
    """
    try:
        # Check CLI availability
        if shutil.which("ollama") is None:
            return {"available": False, "error": "ollama CLI not found on PATH"}

        async def run_cmd(*args):
            # Run CLI command asynchronously and return decoded stdout/stderr
            proc = await _asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await proc.communicate()
            return out.decode(errors="ignore"), err.decode(errors="ignore"), proc.returncode

        list_out, list_err, list_rc = await run_cmd("ollama", "list")
        ps_out, ps_err, ps_rc = await run_cmd("ollama", "ps")

        show_out, show_err, show_rc = (None, None, None)
        if OLLAMA_MODEL:
            show_out, show_err, show_rc = await run_cmd("ollama", "show", OLLAMA_MODEL)

        return {
            "available": True,
            "list": list_out.strip(),
            "list_error": list_err.strip() or None,
            "list_rc": list_rc,
            "ps": ps_out.strip(),
            "ps_error": ps_err.strip() or None,
            "ps_rc": ps_rc,
            "show_model": OLLAMA_MODEL,
            "show": show_out.strip() if show_out else None,
            "show_error": show_err.strip() if show_err else None,
            "show_rc": show_rc,
        }
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Error in /ollama/status")
        return {"available": False, "error": str(e), "traceback": tb}


# ===== Stream Processing Endpoints =====

@app.post("/stream/transcribe")
async def transcribe_stream(
    stream_url: str = Form(..., description="Stream URL (YouTube, Twitch, RTMP, etc.)"),
    quality: str = Form("best", description="Stream quality"),
    chunk_duration: int = Form(30, description="Chunk duration in seconds"),
    translate_to: Optional[str] = Form(None, description="Comma-separated target languages"),
):
    """
    Transcribe a single chunk from a live stream.
    
    - **stream_url**: URL of the stream (YouTube, Twitch, RTMP, HLS, etc.)
    - **quality**: best, worst, 720p, 480p, etc.
    - **chunk_duration**: Duration of chunk to process
    - **translate_to**: Optional translation targets
    """
    try:
        # Download stream chunk
        chunk_file = await stream_service.download_stream_chunk(
            stream_url,
            duration=chunk_duration,
            quality=quality,
        )
        
        if not chunk_file:
            raise HTTPException(status_code=400, detail="Failed to download stream")
        
        try:
            # Transcribe chunk
            segments, metadata = await whisper_service.transcribe(
                str(chunk_file),
                task="transcribe",
            )
            
            full_transcription = " ".join(seg.text for seg in segments)
            
            # Optional translation
            translations = None
            if translate_to:
                target_langs = [lang.strip().lower() for lang in translate_to.split(",")]
                source_lang_map = {"uz": "uzbek", "en": "english", "ru": "russian"}
                source_lang = source_lang_map.get(metadata["language"], metadata["language"])
                
                translations = await llm_service.translate_with_llm(
                    full_transcription,
                    source_lang,
                    target_langs[0] if target_langs else "english",
                ) if target_langs else None
            
            return {
                "success": True,
                "stream_url": stream_url,
                "detected_language": metadata["language"],
                "transcription": full_transcription,
                "translation": translations,
                "duration": metadata["duration"],
            }
            
        finally:
            # Cleanup chunk file
            if chunk_file.exists():
                chunk_file.unlink()
            
    except Exception as e:
        logger.error(f"Stream transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream/info")
async def get_stream_info(url: str):
    """Get information about a stream URL."""
    try:
        qualities = stream_service.get_available_qualities(url)
        
        return {
            "url": url,
            "supported": len(qualities) > 0,
            "qualities": list(qualities.keys()),
        }
    except Exception as e:
        logger.error(f"Stream info error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===== WebSocket for Real-Time Streaming =====

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time stream transcription.
    
    Send: {"action": "start", "stream_url": "...", "quality": "best"}
    Send: {"action": "stop"}
    
    Receive: {"type": "transcription", "text": "...", "language": "..."}
    """
    await manager.connect(websocket)
    
    try:
        stream_active = False
        stream_task = None
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "start":
                if stream_active:
                    await manager.send_message(
                        {"type": "error", "message": "Stream already active"},
                        websocket
                    )
                    continue
                
                stream_url = data.get("stream_url")
                quality = data.get("quality", "best")
                chunk_duration = data.get("chunk_duration", 30)
                
                if not stream_url:
                    await manager.send_message(
                        {"type": "error", "message": "stream_url required"},
                        websocket
                    )
                    continue
                
                # Start streaming
                stream_active = True
                await manager.send_message(
                    {"type": "status", "message": "Stream started"},
                    websocket
                )
                
                # Process stream chunks
                try:
                    async for chunk_file in stream_service.process_stream_continuous(
                        stream_url,
                        quality,
                        chunk_duration
                    ):
                        if not stream_active:
                            break
                        
                        try:
                            # Transcribe chunk
                            segments, metadata = await whisper_service.transcribe(
                                str(chunk_file),
                                task="transcribe",
                            )
                            
                            transcription = " ".join(seg.text for seg in segments)
                            
                            # Send result
                            await manager.send_message({
                                "type": "transcription",
                                "text": transcription,
                                "language": metadata["language"],
                                "timestamp": time.time(),
                            }, websocket)
                            
                        finally:
                            # Cleanup
                            if chunk_file.exists():
                                chunk_file.unlink()
                
                except Exception as e:
                    logger.error(f"Stream processing error: {e}")
                    await manager.send_message(
                        {"type": "error", "message": str(e)},
                        websocket
                    )
                finally:
                    stream_active = False
            
            elif action == "stop":
                stream_active = False
                await manager.send_message(
                    {"type": "status", "message": "Stream stopped"},
                    websocket
                )
            
            elif action == "ping":
                await manager.send_message({"type": "pong"}, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
