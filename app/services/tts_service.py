"""
Text-to-Speech provider abstraction.

Architecture:
  - TTSProvider     → Protocol (interface)
  - OpenAITTS       → OpenAI TTS-1 / TTS-1-HD implementation (streaming + full)
  - ElevenLabsTTS   → Placeholder for ElevenLabs (stub)

All providers expose both `synthesize` (full blob) and `synthesize_stream`
(async generator of audio chunks) to support both voice endpoints.
"""

from collections.abc import AsyncGenerator
from typing import Protocol

from app.clients.openai_client import OpenAIClient
from app.core.exceptions import TTSError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TTSProvider(Protocol):
    """Interface for any text-to-speech provider."""

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> bytes:
        """Return full audio as bytes."""
        ...

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Yield audio chunks progressively."""
        ...


# ---------------------------------------------------------------------------
# OpenAI TTS implementation
# ---------------------------------------------------------------------------

class OpenAITTS:
    """TTS provider backed by OpenAI TTS-1."""

    def __init__(self, client: OpenAIClient) -> None:
        self._client = client

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> bytes:
        """Return the full synthesized audio blob."""
        if not text.strip():
            raise TTSError("Empty text provided for TTS synthesis")
        logger.debug("OpenAITTS.synthesize: %d chars", len(text))
        return await self._client.synthesize(text, voice, response_format)

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Yield audio byte chunks as they arrive from OpenAI."""
        if not text.strip():
            raise TTSError("Empty text provided for streaming TTS")
        logger.debug("OpenAITTS.synthesize_stream: %d chars", len(text))
        async for chunk in self._client.synthesize_stream(text, voice, response_format):
            yield chunk


# ---------------------------------------------------------------------------
# ElevenLabs TTS stub
# ---------------------------------------------------------------------------

class ElevenLabsTTS:
    """
    Placeholder TTS provider for ElevenLabs.

    Implement `synthesize` and `synthesize_stream` with the ElevenLabs SDK.
    Wire it into VoiceService via the tts_provider parameter.
    """

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> bytes:
        raise NotImplementedError("ElevenLabsTTS is a stub.")

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError("ElevenLabsTTS streaming is a stub.")
        yield b""  # makes type-checker happy
