"""Stream processing service for live transcription from streams."""
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, AsyncGenerator
import time

import streamlink

from app.config import (
    STREAM_CHUNK_DURATION,
    STREAM_CHUNKS_DIR,
    MAX_STREAM_DURATION,
)

logger = logging.getLogger(__name__)


class StreamService:
    """Service for processing live streams (RTMP, HLS, YouTube, Twitch, etc.)."""
    
    _active_streams = {}
    
    @staticmethod
    def is_supported_url(url: str) -> bool:
        """Check if URL is supported by streamlink."""
        try:
            streams = streamlink.streams(url)
            return len(streams) > 0
        except Exception as e:
            logger.error(f"Error checking stream URL: {e}")
            return False
    
    @staticmethod
    def get_available_qualities(url: str) -> dict:
        """Get available stream qualities."""
        try:
            return streamlink.streams(url)
        except Exception as e:
            logger.error(f"Error getting stream qualities: {e}")
            return {}
    
    async def download_stream_chunk(
        self,
        url: str,
        duration: int = STREAM_CHUNK_DURATION,
        quality: str = "best",
    ) -> Optional[Path]:
        """Download a chunk of stream to temporary file."""
        try:
            # Get stream
            streams = streamlink.streams(url)
            
            if not streams:
                logger.error(f"No streams found for {url}")
                return None
            
            # Select quality (best, worst, 720p, etc.)
            stream = streams.get(quality) or streams.get("best")
            
            if not stream:
                logger.error(f"Quality '{quality}' not available")
                return None
            
            # Create temp file for chunk
            chunk_file = STREAM_CHUNKS_DIR / f"chunk_{int(time.time())}.ts"
            
            # Download chunk with ffmpeg
            cmd = [
                "ffmpeg",
                "-i", stream.url,
                "-t", str(duration),
                "-c", "copy",
                "-f", "mpegts",
                str(chunk_file),
                "-loglevel", "error",
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                return None
            
            if chunk_file.exists() and chunk_file.stat().st_size > 0:
                return chunk_file
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading stream chunk: {e}")
            return None
    
    async def process_stream_continuous(
        self,
        url: str,
        quality: str = "best",
        chunk_duration: int = STREAM_CHUNK_DURATION,
    ) -> AsyncGenerator[Path, None]:
        """Continuously process stream in chunks."""
        start_time = time.time()
        
        while True:
            # Check max duration
            if time.time() - start_time > MAX_STREAM_DURATION:
                logger.info("Max stream duration reached")
                break
            
            # Download next chunk
            chunk_file = await self.download_stream_chunk(
                url,
                duration=chunk_duration,
                quality=quality,
            )
            
            if chunk_file:
                yield chunk_file
                
                # Wait a bit before next chunk to avoid overlap
                await asyncio.sleep(1)
            else:
                # Stream might be ended or errored
                logger.warning("Failed to get stream chunk, retrying...")
                await asyncio.sleep(5)
    
    async def extract_audio_from_stream(
        self,
        stream_url: str,
        output_path: Path,
        duration: int = 30,
    ) -> bool:
        """Extract audio from live stream."""
        try:
            streams = streamlink.streams(stream_url)
            
            if not streams:
                return False
            
            stream = streams.get("best")
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", stream.url,
                "-t", str(duration),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # WAV format
                "-ar", "16000",  # 16kHz for Whisper
                "-ac", "1",  # Mono
                str(output_path),
                "-y",  # Overwrite
                "-loglevel", "error",
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            await process.communicate()
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    async def get_rtmp_stream_info(self, rtmp_url: str) -> dict:
        """Get information about RTMP stream."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_format",
                "-show_streams",
                "-of", "json",
                rtmp_url,
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                import json
                return json.loads(stdout.decode())
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting stream info: {e}")
            return {}
    
    def cleanup_old_chunks(self, max_age_seconds: int = 3600):
        """Clean up old stream chunks."""
        current_time = time.time()
        
        for chunk_file in STREAM_CHUNKS_DIR.glob("chunk_*.ts"):
            try:
                file_age = current_time - chunk_file.stat().st_mtime
                if file_age > max_age_seconds:
                    chunk_file.unlink()
                    logger.debug(f"Deleted old chunk: {chunk_file.name}")
            except Exception as e:
                logger.error(f"Error deleting chunk: {e}")


# Global service instance
stream_service = StreamService()
