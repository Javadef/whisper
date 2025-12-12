"""Translation service using NLLB model for Uzbek support."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import TRANSLATION_MODEL, LANG_CODES

logger = logging.getLogger(__name__)


class TranslationService:
    """Service for text translation using NLLB-200 model."""
    
    _instance: Optional["TranslationService"] = None
    _model = None
    _tokenizer = None
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls):
        """Get or initialize the translation model (singleton)."""
        if cls._model is None:
            logger.info(f"Loading translation model: {TRANSLATION_MODEL}")
            
            cls._tokenizer = AutoTokenizer.from_pretrained(
                TRANSLATION_MODEL,
                cache_dir="./models",
            )
            
            cls._model = AutoModelForSeq2SeqLM.from_pretrained(
                TRANSLATION_MODEL,
                cache_dir="./models",
                torch_dtype=torch.float16 if cls._device == "cuda" else torch.float32,
            ).to(cls._device)
            
            logger.info(f"Translation model loaded on {cls._device}!")
        
        return cls._model, cls._tokenizer
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded."""
        return cls._model is not None
    
    def _get_lang_code(self, language: str) -> str:
        """Get NLLB language code from language name."""
        lang_lower = language.lower()
        if lang_lower in LANG_CODES:
            return LANG_CODES[lang_lower]
        # Try to find partial match
        for key, code in LANG_CODES.items():
            if key in lang_lower or lang_lower in key:
                return code
        raise ValueError(f"Unsupported language: {language}")
    
    def _translate_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Synchronous translation (runs in thread pool)."""
        model, tokenizer = self.get_model()
        
        src_code = self._get_lang_code(source_lang)
        tgt_code = self._get_lang_code(target_lang)
        
        # Set source language
        tokenizer.src_lang = src_code
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self._device)
        
        # Get target language token ID
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=1024,
                num_beams=5,
                do_sample=False,
            )
        
        # Decode output
        translated_text = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]
        
        return translated_text
    
    async def translate(
        self,
        text: str,
        source_lang: str = "uzbek",
        target_lang: str = "english",
    ) -> str:
        """Async translation using thread pool executor."""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self._executor,
            self._translate_sync,
            text,
            source_lang,
            target_lang,
        )
        
        return result
    
    async def translate_to_multiple(
        self,
        text: str,
        source_lang: str,
        target_langs: list,
    ) -> dict:
        """Translate text to multiple target languages."""
        translations = {}
        
        for target_lang in target_langs:
            try:
                translated = await self.translate(text, source_lang, target_lang)
                translations[target_lang] = translated
            except Exception as e:
                logger.error(f"Translation to {target_lang} failed: {e}")
                translations[target_lang] = f"Error: {str(e)}"
        
        return translations


# Global service instance
translation_service = TranslationService()
