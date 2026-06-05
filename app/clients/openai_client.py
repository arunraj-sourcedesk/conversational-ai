"""
OpenAI async client abstraction layer.

Wraps the official `openai` Python SDK and provides:
- Async chat completion (non-streaming)
- Async chat completion (streaming, yields token deltas)
- Async Whisper STT
- Async TTS synthesis (non-streaming byte blob)
- Async TTS synthesis (streaming, yields audio chunks)
- Retry logic with exponential back-off
- Structured error mapping to domain exceptions
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import openai
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from app.core.config import Settings
from app.core.exceptions import LLMError, STTError, TTSError, TimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """Thin async wrapper around the OpenAI SDK with retry logic."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
            max_retries=settings.openai_max_retries,
        )

    # ------------------------------------------------------------------
    # Chat Completion — non-streaming
    # ------------------------------------------------------------------

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> tuple[str, int]:
        """
        Perform a full (non-streaming) chat completion.

        Returns:
            (reply_text, total_tokens_used)
        """
        model = model or self._settings.openai_chat_model
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = await self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            logger.debug("chat_complete: model=%s tokens=%d", model, tokens)
            return content, tokens

        except APITimeoutError as exc:
            raise TimeoutError("OpenAI request timed out") from exc
        except RateLimitError as exc:
            raise LLMError("OpenAI rate limit exceeded — try again shortly") from exc
        except APIError as exc:
            raise LLMError(f"OpenAI API error: {exc.message}") from exc
        except Exception as exc:
            raise LLMError(f"Unexpected LLM error: {exc}") from exc

    # ------------------------------------------------------------------
    # Chat Completion — streaming
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields token-delta strings as they arrive.

        Usage:
            async for token in client.chat_stream(messages):
                ...
        """
        model = model or self._settings.openai_chat_model
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except APITimeoutError as exc:
            raise TimeoutError("OpenAI stream timed out") from exc
        except RateLimitError as exc:
            raise LLMError("OpenAI rate limit exceeded") from exc
        except APIError as exc:
            raise LLMError(f"OpenAI API error: {exc.message}") from exc
        except Exception as exc:
            raise LLMError(f"Unexpected streaming error: {exc}") from exc

    # ------------------------------------------------------------------
    # Speech-to-Text (Whisper)
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> str:
        """
        Transcribe audio bytes using Whisper.

        Args:
            audio_bytes: Raw audio file content.
            filename: Original filename (used to infer MIME type).
            language: Optional BCP-47 language hint (e.g. "en").

        Returns:
            Transcribed text string.
        """
        import io

        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = filename  # type: ignore[attr-defined]

            kwargs: dict[str, Any] = {
                "model": self._settings.stt_model,
                "file": audio_file,
            }
            if language:
                kwargs["language"] = language

            result = await self._client.audio.transcriptions.create(**kwargs)
            transcript = result.text.strip()
            logger.debug("transcribe: %d chars", len(transcript))
            return transcript

        except APITimeoutError as exc:
            raise TimeoutError("Whisper transcription timed out") from exc
        except APIError as exc:
            raise STTError(f"Whisper API error: {exc.message}") from exc
        except Exception as exc:
            raise STTError(f"Transcription failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Text-to-Speech — non-streaming (full audio blob)
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> bytes:
        """
        Convert text to speech and return the full audio as bytes.

        Returns:
            Raw audio bytes (MP3 or WAV depending on `response_format`).
        """
        voice = voice or self._settings.tts_voice
        response_format = response_format or self._settings.tts_audio_format

        try:
            response = await self._client.audio.speech.create(
                model=self._settings.tts_model,
                voice=voice,  # type: ignore[arg-type]
                input=text,
                response_format=response_format,  # type: ignore[arg-type]
            )
            audio_bytes = response.content
            logger.debug("synthesize: %d bytes, format=%s", len(audio_bytes), response_format)
            return audio_bytes

        except APITimeoutError as exc:
            raise TimeoutError("TTS synthesis timed out") from exc
        except APIError as exc:
            raise TTSError(f"TTS API error: {exc.message}") from exc
        except Exception as exc:
            raise TTSError(f"TTS synthesis failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Text-to-Speech — streaming audio chunks
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio in chunks.

        Yields:
            Raw audio byte chunks.
        """
        voice = voice or self._settings.tts_voice
        response_format = response_format or self._settings.tts_audio_format

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._settings.tts_model,
                voice=voice,  # type: ignore[arg-type]
                input=text,
                response_format=response_format,  # type: ignore[arg-type]
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        yield chunk

        except APITimeoutError as exc:
            raise TimeoutError("TTS stream timed out") from exc
        except APIError as exc:
            raise TTSError(f"TTS stream API error: {exc.message}") from exc
        except Exception as exc:
            raise TTSError(f"TTS stream failed: {exc}") from exc
