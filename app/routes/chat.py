"""
Text chat routes.

Endpoints:
  POST /chat         — non-streaming, returns full JSON response
"""

from fastapi import APIRouter, Depends, Query

from app.core.logging import get_logger
from app.models.chat import (
    ChatHistoryResponse,
    ChatHistoryMessage,
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionMemoryResponse,
    SessionMetadata,
    SessionOutcomeResponse,
)
from app.services.chat_service import ChatService
from app.services.session_store import InMemorySessionStore
from app.utils.dependencies import get_chat_service, get_session_store

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ---------------------------------------------------------------------------
# POST /chat/session  — create a new session
# ---------------------------------------------------------------------------

@router.post(
    "/session",
    response_model=CreateSessionResponse,
    summary="Create a new chat session",
    description="Create a dynamically generated session_id with an optional configured context.",
)
async def create_session(
    body: CreateSessionRequest,
    service: ChatService = Depends(get_chat_service),
) -> CreateSessionResponse:
    """Create a session and optionally configure a system prompt and an initial greeting."""
    session_id = await service.create_session(
        system_prompt=body.system_prompt,
        greeting_message=body.greeting_message,
        client_id=body.client_id,
        lead_id=body.lead_id,
    )
    return CreateSessionResponse(
        session_id=session_id,
        system_prompt=body.system_prompt,
        greeting_message=body.greeting_message,
    )


# ---------------------------------------------------------------------------
# POST /chat  — non-streaming
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    summary="Non-streaming chat completion",
    description="Send a message and receive the full response in one JSON payload.",
)
async def chat(
    body: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Non-streaming text chat.

    - Loads conversation history for `session_id`
    - Calls the LLM
    - Persists the turn
    - Returns the complete reply and optional intent extraction on first turn
    """
    reply, tokens, intent = await service.chat(
        session_id=body.session_id,
        message=body.message,
        system_prompt=body.system_prompt,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    return ChatResponse(
        response=reply,
        session_id=body.session_id,
        tokens_used=tokens,
        intent_extraction=intent,
    )



# ---------------------------------------------------------------------------
# GET /chat/sessions  — list persisted sessions
# ---------------------------------------------------------------------------

@router.get(
    "/sessions",
    response_model=list[SessionMetadata],
    summary="List persisted chat sessions",
    description=(
        "Retrieve stored chat sessions filtered by clientId and/or leadId. "
        "Returns session metadata with created and updated timestamps."
    ),
)
async def list_sessions(
    client_id: str | None = Query(default=None, alias="clientId"),
    lead_id: str | None = Query(default=None, alias="leadId"),
) -> list[SessionMetadata]:
    from app.utils.db import get_sessions
    sessions = await get_sessions(client_id=client_id, lead_id=lead_id)
    return [
        SessionMetadata(
            sessionId=item["session_id"],
            clientId=item["client_id"],
            leadId=item["lead_id"],
            systemPrompt=item.get("system_prompt"),
            greetingMessage=item.get("greeting_message"),
            createdAt=item["created_at"],
            updatedAt=item["updated_at"],
        )
        for item in sessions
    ]


# ---------------------------------------------------------------------------
# GET /chat/sessions/{session_id}/history  — retrieve chat history from the DB
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get chat history from database",
    description="Fetch the chat history for a session, ordered by time and paginated.",
)
async def get_history(
    session_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500, alias="pageSize"),
) -> ChatHistoryResponse:
    """Get chat history for a session."""
    from app.utils.db import get_chat_history, get_session

    page = max(1, page)
    page_size = min(max(1, page_size), 500)
    history = await get_chat_history(session_id, page=page, page_size=page_size)
    session_data = await get_session(session_id)
    messages = [
        ChatHistoryMessage(
            sender="AI" if item["role"] == "assistant" else "USER",
            message=item["message"],
            timestamp=item["created_at"],
            tokens_used=item.get("tokens_used"),
            intent_extraction=item.get("intent_extraction"),
        )
        for item in history
    ]
    return ChatHistoryResponse(
        session_id=session_id,
        system_prompt=session_data.get("system_prompt") if session_data else None,
        greeting_message=session_data.get("greeting_message") if session_data else None,
        page=page,
        page_size=page_size,
        messages=messages,
    )


@router.get(
    "/sessions/{session_id}/outcome",
    response_model=SessionOutcomeResponse,
    summary="Summarize the outcome of a session",
    description="Use the chat history for a session and an LLM to produce a structured outcome summary.",
)
async def get_session_outcome(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> SessionOutcomeResponse:
    """Generate a structured outcome summary for the session."""
    return await service.get_session_outcome(session_id)


@router.get(
    "/sessions/{session_id}/memory",
    response_model=SessionMemoryResponse,
    summary="Inspect in-memory session state",
    description="Return the current in-memory conversation history and configured system prompt for a session.",
)
async def get_session_memory(
    session_id: str,
    store: InMemorySessionStore = Depends(get_session_store),
) -> SessionMemoryResponse:
    """Get current session-store data for a session."""
    from app.utils.db import get_session

    history = await store.get_history(session_id)
    system_prompt = await store.get_system_prompt(session_id)
    session_data = await get_session(session_id)
    return SessionMemoryResponse(
        session_id=session_id,
        system_prompt=system_prompt,
        greeting_message=session_data.get("greeting_message") if session_data else None,
        messages=history,
    )
