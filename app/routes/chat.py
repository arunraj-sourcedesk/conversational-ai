"""
Text chat routes.

Endpoints:
  POST /chat         — non-streaming, returns full JSON response
  POST /chat/stream  — SSE streaming, yields token deltas
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.logging import get_logger
from app.models.chat import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    CreateSessionResponse,
)
from app.services.chat_service import ChatService
from app.utils.dependencies import get_chat_service

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
        session_id=body.session_id,
        system_prompt=body.system_prompt,
        greeting_message=body.greeting_message,
    )
    return CreateSessionResponse(session_id=session_id)


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
# POST /chat/stream  — SSE streaming
# ---------------------------------------------------------------------------

@router.post(
    "/stream",
    summary="Streaming chat via Server-Sent Events",
    description=(
        "Stream token deltas in real time using SSE (text/event-stream). "
        "Each event is a JSON object: `{\"delta\": \"...\", \"done\": false}`. "
        "A final `{\"delta\": \"\", \"done\": true}` signals end-of-stream."
    ),
    response_class=StreamingResponse,
)
async def chat_stream(
    body: ChatRequest,
    request: Request,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Streaming text chat via Server-Sent Events.

    Clients should consume the event stream and concatenate `delta` values
    until they receive a chunk with `done: true`.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for token in service.chat_stream(
                session_id=body.session_id,
                message=body.message,
                system_prompt=body.system_prompt,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                # Check if the client has disconnected
                if await request.is_disconnected():
                    logger.info("SSE client disconnected session=%s", body.session_id)
                    break

                payload = json.dumps({"delta": token, "done": False})
                yield f"data: {payload}\n\n"

            # End-of-stream marker
            yield f"data: {json.dumps({'delta': '', 'done': True})}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled session=%s", body.session_id)
        except Exception as exc:
            logger.exception("SSE stream error session=%s: %s", body.session_id, exc)
            error_payload = json.dumps({"error": str(exc), "done": True})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering
            "Connection": "keep-alive",
        },
    )
