"""
Speech-to-Text provider abstraction.

Architecture:
  - STTProvider   → Protocol (interface)
  - WhisperSTT    → OpenAI Whisper implementation
  - ElevenLabsSTT → Placeholder for ElevenLabs (stub)

To add a new STT backend, implement STTProvider and wire it
into VoiceService via dependency injection.
"""

from typing import Protocol

from app.clients.openai_client import OpenAIClient
from app.core.exceptions import STTError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class STTProvider(Protocol):
    """Interface for any speech-to-text provider."""

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> str:
        """Return transcribed text from raw audio bytes."""
        ...


# ---------------------------------------------------------------------------
# OpenAI Whisper implementation
# ---------------------------------------------------------------------------

class WhisperSTT:
    """STT provider backed by OpenAI Whisper."""

    def __init__(self, client: OpenAIClient) -> None:
        self._client = client

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> str:
        """Delegate transcription to the OpenAI client wrapper."""
        if not audio_bytes:
            raise STTError("Empty audio payload received")
        logger.debug("WhisperSTT.transcribe: %d bytes", len(audio_bytes))
        return await self._client.transcribe(audio_bytes, filename, language)


# ---------------------------------------------------------------------------
# ElevenLabs STT stub (swap in when ready)
# ---------------------------------------------------------------------------

class ElevenLabsSTT:
    """
    Placeholder STT provider for ElevenLabs.

    Replace the body of `transcribe` with the real ElevenLabs SDK call.
    The rest of the system requires no other changes.
    """

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> str:
        raise NotImplementedError(
            "ElevenLabsSTT is a stub — implement with the ElevenLabs SDK."
        )
