"""Pydantic models for request/response schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class TargetLanguage(str, Enum):
    ENGLISH = "english"
    RUSSIAN = "russian"


class TranscriptionTask(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"  # Whisper's built-in translation to English


class TranscriptionRequest(BaseModel):
    """Request model for transcription settings."""
    task: TranscriptionTask = TranscriptionTask.TRANSCRIBE
    language: Optional[str] = Field(None, description="Source language code (e.g., 'uz', 'en'). Auto-detected if not provided.")
    translate_to: Optional[List[TargetLanguage]] = Field(None, description="Target languages for translation")
    word_timestamps: bool = Field(False, description="Include word-level timestamps")
    

class Segment(BaseModel):
    """A transcription segment."""
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[dict]] = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription results."""
    success: bool
    filename: str
    detected_language: str
    language_probability: float
    duration: float
    transcription: str
    segments: List[Segment]
    translations: Optional[dict] = None
    processing_time: float


class TranslationRequest(BaseModel):
    """Request for text translation."""
    text: str
    source_language: str = "uzbek"
    target_language: str = "english"


class TranslationResponse(BaseModel):
    """Response for text translation."""
    success: bool
    source_text: str
    translated_text: str
    source_language: str
    target_language: str


class TaskStatus(BaseModel):
    """Status of an async task."""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = None
    result: Optional[TranscriptionResponse] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    models_loaded: bool
    llm_loaded: bool = False
    whisper_model: str = ""
    llm_model: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request."""
    message: str
    session_id: str = "default"
    context: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response."""
    response: str
    session_id: str


class StreamTranscribeRequest(BaseModel):
    """Stream transcription request."""
    stream_url: str
    quality: str = "best"
    chunk_duration: int = 30
    translate_to: Optional[List[str]] = None
    continuous: bool = False


class StreamChunkResult(BaseModel):
    """Result for a stream chunk."""
    chunk_id: int
    timestamp: float
    transcription: str
    detected_language: str
    translations: Optional[dict] = None


class SummarizeRequest(BaseModel):
    """Summarization request."""
    text: str
    language: str = "english"


class LLMTranslateRequest(BaseModel):
    """LLM-based translation request."""
    text: str
    source_language: str
    target_language: str
