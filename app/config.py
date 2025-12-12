"""Configuration settings for the Whisper API."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"  # Local models storage
STREAM_CHUNKS_DIR = BASE_DIR / "stream_chunks"

# Create directories if they don't exist
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR, STREAM_CHUNKS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Whisper Model Settings (optimized for RTX 4060 8GB VRAM)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "turbo")  # Fast and accurate (809MB)
WHISPER_DEVICE = "cuda"  # Use GPU with cuDNN installed
WHISPER_COMPUTE_TYPE = "int8_float16"  # Faster inference with minimal quality loss

# Alternative compute types:
# "int8_float16" - Faster, less VRAM, slightly lower quality
# "int8" - Fastest, lowest VRAM usage
# "float16" - Best quality for consumer GPUs

# Translation Model (for Uzbek translations)
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"  # Supports Uzbek

# LLM Settings (for Q&A and better translations)
# Use local Meta Llama 3 8B GGUF model if present in the models/ folder
LLM_MODEL_PATH = MODELS_DIR / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"  # Download to models folder
LLM_CONTEXT_SIZE = 4096  # Reduced for faster inference
LLM_GPU_LAYERS = 35  # Offload all layers to GPU for RTX 4060 (8B model needs more)
LLM_THREADS = 8
LLM_BATCH_SIZE = 512

# Alternative models (place in models folder):
# - llama-3.2-3b-instruct.Q4_K_M.gguf (3B, best for RTX 4060, great translations)
# - qwen2.5-7b-instruct.Q4_K_M.gguf (7B, better multilingual, needs more VRAM)
# - gemma-2-9b-it.Q4_K_M.gguf (9B, excellent quality, 5GB VRAM)

# Language codes for NLLB model
LANG_CODES = {
    "uzbek": "uzn_Latn",      # Uzbek (Latin script)
    "english": "eng_Latn",    # English
    "russian": "rus_Cyrl",    # Russian
    "uz": "uzn_Latn",
    "en": "eng_Latn", 
    "ru": "rus_Cyrl",
}

# Supported audio/video formats
SUPPORTED_FORMATS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",  # Video
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",  # Audio
}

# Max file size (500MB)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Concurrent processing limit
MAX_CONCURRENT_TASKS = 3

# Streaming settings
STREAM_CHUNK_DURATION = 30  # seconds
STREAM_BUFFER_SIZE = 1024 * 1024  # 1MB
MAX_STREAM_DURATION = 3600  # 1 hour max

# WebSocket settings
WS_HEARTBEAT_INTERVAL = 30  # seconds
WS_MAX_CONNECTIONS = 10

# Chat settings
CHAT_MAX_HISTORY = 20
CHAT_TEMPERATURE = 0.7
CHAT_MAX_TOKENS = 2048

# Ollama (optional local or cloud Ollama server)
# Default host for local Ollama is http://localhost:11434
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Default model name to use with Ollama (override with env var OLLAMA_MODEL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
# If set to 'true', prefer Ollama (cloud/local) over local llama-cpp model even if GGUF exists
# Set to 'false' to use local GGUF model (Meta-Llama-3-8B) when available
OLLAMA_PREFERRED = os.getenv("OLLAMA_PREFERRED", "false").lower()
