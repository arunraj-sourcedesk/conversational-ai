"""
Pydantic models for text-based chat endpoints.
Strict typing ensures clean validation and OpenAPI schema generation.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    """Payload for POST /chat/session."""

    model_config = {
        "populate_by_name": True
    }

    client_id: str = Field(
        ..., 
        alias="clientId",
        description="Client identifier associated with this session.",
    )
    lead_id: str = Field(
        ..., 
        alias="leadId",
        description="Lead identifier associated with this session.",
    )
    system_prompt: str | None = Field(
        default=None,
        alias="systemPrompt",
        description="Configured context (system prompt) for this session to be used throughout the conversation.",
    )
    greeting_message: str | None = Field(
        default=None,
        alias="greetingMessage",
        description="The initial AI greeting message for the session.",
    )


class ChatRequest(BaseModel):
    """Payload for POST /chat and POST /chat/stream."""

    model_config = {
        "populate_by_name": True
    }

    message: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="The user's input message.",
        examples=["Hello, how are you?"],
    )
    session_id: str = Field(
        ...,
        alias="sessionId",
        min_length=1,
        max_length=128,
        description="Unique identifier for the conversation session.",
        examples=["abc123"],
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=4096,
        description="Optional system prompt override for this turn.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate.",
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class IntentExtraction(BaseModel):
    """Extracted intent from the user's first response."""

    permission_to_continue: bool = Field(default=False)
    identity_confirmed: bool = Field(default=False)
    bad_time_to_talk: bool = Field(default=False)
    wrong_person: bool = Field(default=False)
    opted_out: bool = Field(default=False)
    lead_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)


class CreateSessionResponse(BaseModel):
    """Response for POST /chat/session."""

    model_config = {"populate_by_name": True}

    session_id: str = Field(..., alias="sessionId", description="Session ID.")


class SessionMetadata(BaseModel):
    """Session summary returned by GET /chat/sessions."""

    model_config = {"populate_by_name": True}

    session_id: str = Field(..., alias="sessionId")
    client_id: str = Field(..., alias="clientId")
    lead_id: str = Field(..., alias="leadId")
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")


class ChatHistoryMessage(BaseModel):
    """Single chat message returned by history APIs."""

    sender: str = Field(..., description="USER or AI")
    message: str = Field(...)
    timestamp: str = Field(...)
    tokens_used: int | None = Field(default=None, description="Tokens consumed (assistant rows only).")
    intent_extraction: IntentExtraction | None = Field(
        default=None, description="Extracted intent from the first user turn (assistant rows only)."
    )


class ChatHistoryResponse(BaseModel):
    """Response for GET /chat/sessions/{sessionId}/history."""

    model_config = {"populate_by_name": True}

    session_id: str = Field(..., alias="sessionId")
    page: int = Field(...)
    page_size: int = Field(..., alias="pageSize")
    messages: list[ChatHistoryMessage]


class ConversationMessage(BaseModel):
    """A single turn stored in session memory."""

    role: str = Field(..., description="'user' or 'assistant'.")
    content: str = Field(..., description="Message content.")


class SessionMemoryResponse(BaseModel):
    """Response for GET /chat/sessions/{sessionId}/memory."""

    model_config = {"populate_by_name": True}

    session_id: str = Field(..., alias="sessionId")
    system_prompt: str | None = Field(
        default=None,
        alias="systemPrompt",
        description="Configured system prompt for this in-memory session.",
    )
    messages: list[ConversationMessage]


class ChatResponse(BaseModel):
    """Response for POST /chat (non-streaming)."""

    response: str = Field(..., description="The assistant's reply.")
    session_id: str = Field(..., description="Echo of the session ID.")
    tokens_used: int | None = Field(
        default=None, description="Total tokens consumed (prompt + completion)."
    )
    intent_extraction: IntentExtraction | None = Field(
        default=None, description="Extracted intent if this was the first turn."
    )


class StreamChunk(BaseModel):
    """A single SSE data chunk for POST /chat/stream."""

    delta: str = Field(..., description="Partial token text.")
    done: bool = Field(default=False, description="True on the final chunk.")
