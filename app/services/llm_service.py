"""LLM service for Q&A and advanced translations using llama.cpp."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict
import logging
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from app.config import (
    LLM_MODEL_PATH,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    LLM_THREADS,
    LLM_BATCH_SIZE,
    CHAT_TEMPERATURE,
    CHAT_MAX_TOKENS,
    OLLAMA_PREFERRED,
)
# Optional Ollama service fallback
try:
    from app.services.ollama_service import ollama_service
except Exception:
    ollama_service = None

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based Q&A and translation."""
    
    _instance: Optional["LLMService"] = None
    _model: Optional["Llama"] = None
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)
    _conversation_history: Dict[str, List[Dict]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls) -> Optional["Llama"]:
        """Get or initialize the LLM model (singleton)."""
        if Llama is None:
            logger.warning("llama-cpp-python not installed. LLM features disabled.")
            return None
            
        if cls._model is None:
            if not LLM_MODEL_PATH.exists():
                logger.warning(f"LLM model not found at {LLM_MODEL_PATH}")
                logger.info("Download a model from https://huggingface.co/models?library=gguf")
                logger.info("Recommended: Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
                return None
            
            logger.info(f"Loading LLM model: {LLM_MODEL_PATH.name}")
            
            try:
                cls._model = Llama(
                    model_path=str(LLM_MODEL_PATH),
                    n_ctx=LLM_CONTEXT_SIZE,
                    n_gpu_layers=LLM_GPU_LAYERS,
                    n_threads=LLM_THREADS,
                    n_batch=LLM_BATCH_SIZE,
                    verbose=False,
                    use_mmap=True,  # Memory-mapped for faster loading
                    use_mlock=False,  # Don't lock in RAM
                )
                logger.info("LLM model loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")
                return None
        
        return cls._model
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded."""
        return cls._model is not None
    
    def _generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = CHAT_TEMPERATURE,
        max_tokens: int = CHAT_MAX_TOKENS,
    ) -> str:
        """Synchronous text generation."""
        model = self.get_model()
        if model is None:
            return "LLM not available. Please install model."
        
        # Format with system prompt if provided
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        response = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>", "<|im_end|>", "<|eot_id|>"],
        )
        
        return response["choices"][0]["message"]["content"]
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = CHAT_TEMPERATURE,
        max_tokens: int = CHAT_MAX_TOKENS,
    ) -> str:
        """Async text generation."""
        # Try local GGUF model first, fall back to Ollama if not available
        local_model = self.get_model()
        
        if local_model is not None and OLLAMA_PREFERRED != "true":
            # Use local llama-cpp model
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._generate_sync,
                prompt,
                system_prompt,
                temperature,
                max_tokens,
            )
            return result
        
        # Fall back to Ollama if local model not available or Ollama preferred
        if ollama_service is not None:
            # Use Ollama async chat (better than generate for instruction models)
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            return await ollama_service.chat(full_prompt, model=None, stream=False)
        
        return "LLM not available. Please install a model or configure Ollama."
    
    async def chat(
        self,
        message: str,
        session_id: str = "default",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Chat with conversation history."""
        # Initialize history for new sessions
        if session_id not in self._conversation_history:
            self._conversation_history[session_id] = []
        
        history = self._conversation_history[session_id]
        
        # Build prompt with history
        context = ""
        if system_prompt:
            context += f"System: {system_prompt}\n\n"
        
        # Add recent history
        for msg in history[-10:]:  # Last 10 messages
            context += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        context += f"User: {message}\nAssistant:"
        
        # Generate response
        response = await self.generate(context, temperature=0.7, max_tokens=1024)
        
        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Trim history if too long
        if len(history) > 40:
            self._conversation_history[session_id] = history[-40:]
        
        return response
    
    async def translate_with_llm(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Advanced translation using LLM."""
        system_prompt = f"""You are a professional translator. Translate the following text from {source_lang} to {target_lang}.
Maintain the original meaning, tone, and style. Provide only the translation without explanations."""
        
        prompt = f"Translate this {source_lang} text to {target_lang}:\n\n{text}"
        
        translation = await self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more accurate translation
            max_tokens=2048,
        )
        
        return translation.strip()
    
    async def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> str:
        """Answer a question, optionally with context (e.g., from transcription)."""
        if context:
            system_prompt = f"""You are a helpful assistant. Answer questions based on the following context.
If the answer is not in the context, say so.

Context:
{context[:4000]}  # Limit context size
"""
            prompt = question
        else:
            system_prompt = "You are a helpful, knowledgeable assistant."
            prompt = question
        
        answer = await self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1024,
        )
        
        return answer
    
    async def summarize_transcription(
        self,
        transcription: str,
        language: str = "english",
    ) -> str:
        """Summarize a transcription."""
        system_prompt = f"You are a summarization expert. Provide a concise summary in {language}."
        
        prompt = f"Summarize the following transcription:\n\n{transcription[:8000]}"
        
        summary = await self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=512,
        )
        
        return summary.strip()
    
    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        if session_id in self._conversation_history:
            del self._conversation_history[session_id]


# Global service instance
llm_service = LLMService()
