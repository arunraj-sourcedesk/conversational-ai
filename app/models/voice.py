"""
Pydantic models for voice/audio endpoints.
"""

from pydantic import BaseModel, Field


class VoiceChatMetadata(BaseModel):
    """
    JSON metadata returned alongside audio in multipart responses,
    or embedded in the response headers for streaming.
    """

    session_id: str = Field(..., description="Session identifier.")
    transcription: str = Field(..., description="STT transcript of the user's audio.")
    response_text: str = Field(..., description="LLM response text before TTS synthesis.")
    audio_format: str = Field(..., description="MIME type of the returned audio, e.g. audio/mpeg.")
    duration_ms: int | None = Field(
        default=None, description="Estimated TTS audio duration in milliseconds."
    )


class VoiceStreamChunk(BaseModel):
    """Describes a single audio chunk in a streaming voice response."""

    chunk_index: int = Field(..., description="Zero-based chunk sequence number.")
    done: bool = Field(default=False, description="True on the final chunk.")
