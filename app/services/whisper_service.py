"""Faster Whisper transcription service."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple
import logging
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)
from app.models import Segment

logger = logging.getLogger(__name__)


class WhisperService:
    """Service for audio/video transcription using Faster Whisper."""
    
    _instance: Optional["WhisperService"] = None
    _model: Optional[WhisperModel] = None
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls) -> WhisperModel:
        """Get or initialize the Whisper model (singleton)."""
        if cls._model is None:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
            logger.info(f"Device: {WHISPER_DEVICE}, Compute type: {WHISPER_COMPUTE_TYPE}")
            
            cls._model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                download_root="./models",
                # Optimization settings for RTX 4060
                num_workers=4,  # Increased for faster processing
                cpu_threads=8,  # Use more CPU threads
            )
            logger.info("Whisper model loaded successfully!")
        return cls._model
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded."""
        return cls._model is not None
    
    def _transcribe_sync(
        self,
        audio_path: str,
        task: str = "transcribe",
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Tuple[List[Segment], dict]:
        """Synchronous transcription (runs in thread pool)."""
        model = self.get_model()
        
        # Transcription options tuned for lower latency:
        # - single beam / no best_of to avoid multiple-pass decoding
        # - temperature fallback to handle repetition loops
        # - VAD enabled to skip silence
        # - condition_on_previous_text=False to prevent repetition loops
        segments_gen, info = model.transcribe(
            audio_path,
            task=task,
            language=language,
            beam_size=5,  # Increased for better accuracy
            best_of=1,
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.1,  # Penalize repeated phrases
            no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Fallback temperatures for repetition
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,  # Disable to prevent repetition loops
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
        )
        
        # Convert generator to list of segments
        segments = []
        for seg in segments_gen:
            segment = Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=[
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (seg.words or [])
                ] if word_timestamps and seg.words else None
            )
            segments.append(segment)
        
        metadata = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }
        
        return segments, metadata
    
    async def transcribe(
        self,
        audio_path: str,
        task: str = "transcribe",
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Tuple[List[Segment], dict]:
        """Async transcription using thread pool executor."""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self._executor,
            self._transcribe_sync,
            audio_path,
            task,
            language,
            word_timestamps,
        )
        
        return result


# Global service instance
whisper_service = WhisperService()
