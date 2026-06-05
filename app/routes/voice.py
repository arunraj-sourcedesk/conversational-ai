"""
Voice routes.

Endpoints:
  POST /voice/chat        — non-streaming: audio in, audio file out
  POST /voice/chat/stream — streaming: audio in, chunked audio out
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.core.config import Settings, get_settings
from app.core.exceptions import AudioProcessingError
from app.core.logging import get_logger
from app.services.voice_service import VoiceService
from app.utils.audio import get_audio_mime_type, validate_audio_bytes
from app.utils.dependencies import get_voice_service

logger = get_logger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice"])


# ---------------------------------------------------------------------------
# POST /voice/chat  — non-streaming
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    summary="Non-streaming voice chat",
    description=(
        "Upload an audio file (WAV, MP3, etc.), receive a synthesized audio reply. "
        "Response headers include `X-Transcript` and `X-Response-Text` for debugging."
    ),
)
async def voice_chat(
    request: Request,
    audio: UploadFile = File(..., description="Audio recording of the user's speech."),
    session_id: str = Form(..., description="Unique session identifier."),
    language: str | None = Form(default=None, description="BCP-47 language hint (e.g. 'en')."),
    service: VoiceService = Depends(get_voice_service),
    settings: Settings = Depends(get_settings),
) -> Response:
    """
    Full voice pipeline (non-streaming).

    1. Read uploaded audio
    2. STT → transcript
    3. LLM → text response
    4. TTS → audio bytes
    5. Return audio with metadata in headers
    """
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.wav"

    validate_audio_bytes(audio_bytes, settings.max_audio_size_mb, filename)

    audio_out, transcript, response_text = await service.process(
        session_id=session_id,
        audio_bytes=audio_bytes,
        filename=filename,
        language=language,
    )

    mime = get_audio_mime_type(f".{settings.tts_audio_format}")
    logger.info(
        "voice_chat session=%s transcript_len=%d audio_out=%d",
        session_id, len(transcript), len(audio_out),
    )

    return Response(
        content=audio_out,
        media_type=mime,
        headers={
            "X-Transcript": transcript[:200],         # truncated for header safety
            "X-Response-Text": response_text[:200],
            "X-Session-Id": session_id,
        },
    )


# ---------------------------------------------------------------------------
# POST /voice/chat/stream  — streaming audio response
# ---------------------------------------------------------------------------

@router.post(
    "/chat/stream",
    summary="Streaming voice chat",
    description=(
        "Upload an audio file. Receive a chunked audio stream (Transfer-Encoding: chunked). "
        "Suitable for real-time playback as chunks arrive."
    ),
    response_class=StreamingResponse,
)
async def voice_chat_stream(
    request: Request,
    audio: UploadFile = File(..., description="Audio recording of the user's speech."),
    session_id: str = Form(..., description="Unique session identifier."),
    language: str | None = Form(default=None, description="BCP-47 language hint."),
    service: VoiceService = Depends(get_voice_service),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Streaming voice pipeline.

    The response body is a continuous stream of raw audio bytes using
    chunked transfer encoding. Clients can begin playback immediately
    as the first chunks arrive.

    Pipeline:
      Upload → STT (blocks) → LLM (full collect) → TTS stream → chunks
    """
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.wav"

    validate_audio_bytes(audio_bytes, settings.max_audio_size_mb, filename)

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in service.process_stream(
                session_id=session_id,
                audio_bytes=audio_bytes,
                filename=filename,
                language=language,
            ):
                if await request.is_disconnected():
                    logger.info("voice_stream client disconnected session=%s", session_id)
                    break
                yield chunk
        except asyncio.CancelledError:
            logger.info("voice_stream cancelled session=%s", session_id)
        except Exception as exc:
            logger.exception("voice_stream error session=%s: %s", session_id, exc)
            # Can't send a JSON error in a binary audio stream;
            # the stream simply ends. The client must detect silence / short audio.

    mime = get_audio_mime_type(f".{settings.tts_audio_format}")

    return StreamingResponse(
        audio_generator(),
        media_type=mime,
        headers={
            "X-Session-Id": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
