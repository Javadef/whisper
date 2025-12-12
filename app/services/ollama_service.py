"""Ollama service wrapper for chat and generation using `ollama` Python client."""
import asyncio
import logging
from typing import Optional, List, Dict

try:
    from ollama import AsyncClient, Client, chat
except Exception:
    AsyncClient = None
    Client = None
    chat = None

from app.config import OLLAMA_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class OllamaService:
    """Async wrapper around the Ollama client.

    Provides simple `chat` and `generate` methods similar to other LLM
    services in this project.
    """

    _instance: Optional["OllamaService"] = None
    _client: Optional[AsyncClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self) -> Optional[AsyncClient]:
        if AsyncClient is None:
            logger.warning("`ollama` package not installed. Ollama disabled.")
            return None

        if self._client is None:
            self._client = AsyncClient(host=OLLAMA_URL)

        return self._client

    async def chat(self, message: str, model: str = None, stream: bool = False) -> str:
        """Send a chat-style message and return the assistant response.

        If `stream=True`, this will collect streamed parts and concatenate them.
        """
        model = model or OLLAMA_MODEL
        client = self._get_client()
        if client is None:
            return "Ollama client not available."

        messages = [{"role": "user", "content": message}]

        try:
            if stream:
                parts = []
                async for part in client.chat(model=model, messages=messages, stream=True):
                    # part may be dict-like or object
                    try:
                        content = part["message"]["content"]
                    except Exception:
                        try:
                            content = part.message.content
                        except Exception:
                            content = str(part)
                    parts.append(content)
                return "".join(parts)
            else:
                resp = await client.chat(model=model, messages=messages)
                try:
                    return resp["message"]["content"]
                except Exception:
                    try:
                        return resp.message.content
                    except Exception:
                        return str(resp)
        except Exception as e:
            logger.exception("Ollama chat error")
            return f"Ollama error: {e}"

    async def generate(self, prompt: str, model: str = None, stream: bool = False) -> str:
        """Generate from a prompt (non-chat)."""
        model = model or OLLAMA_MODEL
        client = self._get_client()
        if client is None:
            return "Ollama client not available."

        try:
            if stream:
                parts = []
                async for part in client.generate(model=model, prompt=prompt, stream=True):
                    try:
                        content = part["message"]["content"]
                    except Exception:
                        try:
                            content = part.message.content
                        except Exception:
                            content = str(part)
                    parts.append(content)
                return "".join(parts)
            else:
                resp = await client.generate(model=model, prompt=prompt)
                try:
                    return resp["message"]["content"]
                except Exception:
                    try:
                        return resp.message.content
                    except Exception:
                        return str(resp)
        except Exception as e:
            logger.exception("Ollama generate error")
            return f"Ollama error: {e}"


# Global instance
ollama_service = OllamaService()
