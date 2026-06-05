"""
Audio utility helpers.

Provides:
- MIME type resolution from filename
- Audio byte validation
- Temporary file cleanup context manager
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from app.core.exceptions import AudioProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Supported audio MIME types
AUDIO_MIME_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def get_audio_mime_type(filename: str) -> str:
    """Return the MIME type for a given audio filename."""
    ext = Path(filename).suffix.lower()
    mime = AUDIO_MIME_TYPES.get(ext)
    if not mime:
        raise AudioProcessingError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(AUDIO_MIME_TYPES.keys())}"
        )
    return mime


def validate_audio_bytes(
    audio_bytes: bytes,
    max_size_mb: int = 25,
    filename: str = "audio.wav",
) -> None:
    """
    Validate raw audio bytes.

    Raises:
        AudioProcessingError: if the payload is empty or too large.
    """
    if not audio_bytes:
        raise AudioProcessingError("Audio payload is empty")

    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise AudioProcessingError(
            f"Audio file size {size_mb:.1f} MB exceeds maximum of {max_size_mb} MB"
        )

    # Basic format check — validate the extension is supported
    get_audio_mime_type(filename)

    logger.debug("validate_audio: %.2f MB, filename=%s", size_mb, filename)


@asynccontextmanager
async def temp_audio_file(audio_bytes: bytes, suffix: str = ".wav"):
    """
    Async context manager that writes audio bytes to a temp file,
    yields the path, and cleans up on exit.

    Usage:
        async with temp_audio_file(data, ".mp3") as path:
            process(path)
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        logger.debug("temp_audio_file: created %s", tmp.name)
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
            logger.debug("temp_audio_file: deleted %s", tmp.name)
        except OSError:
            pass
