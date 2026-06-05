"""
Voice service — the full audio-in / audio-out pipeline.

Pipeline for non-streaming:
  Audio bytes → STT → LLM (full) → TTS → Audio bytes

Pipeline for streaming:
  Audio bytes → STT → LLM (stream) → TTS stream → Audio chunks

Parallelism note:
  STT must complete before LLM can start (we need the transcript).
  LLM streaming output is piped directly to TTS streaming input
  via an asyncio.Queue, minimising end-to-end latency.
"""

import asyncio
from collections.abc import AsyncGenerator

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.chat_service import ChatService
from app.services.stt_service import STTProvider
from app.services.tts_service import TTSProvider

logger = get_logger(__name__)


class VoiceService:
    """Orchestrates the complete voice pipeline."""

    def __init__(
        self,
        stt: STTProvider,
        chat_service: ChatService,
        tts: TTSProvider,
        settings: Settings,
    ) -> None:
        self._stt = stt
        self._chat = chat_service
        self._tts = tts
        self._settings = settings

    # ------------------------------------------------------------------
    # Non-streaming: Audio → Audio
    # ------------------------------------------------------------------

    async def process(
        self,
        session_id: str,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> tuple[bytes, str, str]:
        """
        Full non-streaming voice pipeline.

        Returns:
            (audio_bytes, transcript, response_text)
        """
        # Step 1: STT
        transcript = await self._stt.transcribe(audio_bytes, filename, language)
        logger.info("voice.process session=%s transcript_len=%d", session_id, len(transcript))

        # Step 2: LLM
        response_text, _ = await self._chat.chat(session_id=session_id, message=transcript)
        logger.info("voice.process session=%s response_len=%d", session_id, len(response_text))

        # Step 3: TTS
        audio_out = await self._tts.synthesize(response_text)
        logger.info("voice.process session=%s audio_bytes=%d", session_id, len(audio_out))

        return audio_out, transcript, response_text

    # ------------------------------------------------------------------
    # Streaming: Audio → streamed Audio chunks
    # ------------------------------------------------------------------

    async def process_stream(
        self,
        session_id: str,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Streaming voice pipeline.

        Approach:
          1. Block on STT (must complete first)
          2. Collect full LLM reply (accumulate streamed tokens)
          3. Stream TTS audio chunks

        For ultra-low latency you could sentence-segment the LLM stream
        and pipe each sentence independently to TTS. That optimisation is
        left as a production extension point here.

        Yields:
            Raw audio byte chunks.
        """
        # Step 1: STT (blocking — we need the transcript before LLM)
        transcript = await self._stt.transcribe(audio_bytes, filename, language)
        logger.info("voice.stream session=%s transcript_len=%d", session_id, len(transcript))

        # Step 2: Collect full LLM response
        # (Sentence-level streaming TTS is a future optimisation)
        llm_tokens: list[str] = []
        async for token in self._chat.chat_stream(session_id=session_id, message=transcript):
            llm_tokens.append(token)

        full_response = "".join(llm_tokens)
        logger.info("voice.stream session=%s response_len=%d", session_id, len(full_response))

        # Step 3: Stream TTS
        async for audio_chunk in self._tts.synthesize_stream(full_response):
            yield audio_chunk
