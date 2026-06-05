"""
Chat service layer.

Responsibility:
  - Load conversation history from the session store
  - Build the OpenAI messages array (system + history + new user message)
  - Call the OpenAI client (streaming or non-streaming)
  - Persist the assistant's reply to session memory
  - Return structured results to the route layer

This module is the single authoritative place where conversation
context is assembled; routes must not build raw OpenAI message dicts.
"""

from collections.abc import AsyncGenerator

from app.clients.openai_client import OpenAIClient
from app.core.config import Settings
from app.core.logging import get_logger
from app.models.chat import ConversationMessage, IntentExtraction
from app.services.session_store import InMemorySessionStore

logger = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, persuasive, and friendly human sales and support representative. "
    "You must never disclose that you are an AI, a program, or a bot. Always maintain a natural, conversational human persona. "
    "Respond in plain text unless the user explicitly asks for formatting."
)


class ChatService:
    """Orchestrates conversation context, LLM calls, and session persistence."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        session_store: InMemorySessionStore,
        settings: Settings,
    ) -> None:
        self._client = openai_client
        self._sessions = session_store
        self._settings = settings

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    async def create_session(
        self, session_id: str | None = None, system_prompt: str | None = None, greeting_message: str | None = None
    ) -> str:
        """Create a new session and return the generated session ID."""
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        await self._sessions.create_session(session_id, system_prompt)
        
        if greeting_message:
            await self._sessions.append(
                session_id, ConversationMessage(role="assistant", content=greeting_message)
            )

        logger.info("create_session session=%s", session_id)
        return session_id

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def chat(
        self,
        session_id: str,
        message: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, int, IntentExtraction | None]:
        """
        Full (non-streaming) chat turn.

        Returns:
            (assistant_reply, tokens_used, intent_extraction)
        """
        history = await self._sessions.get_history(session_id)
        is_first_turn = not any(msg.role == "user" for msg in history)
        
        intent = None
        if is_first_turn:
            intent = await self._extract_intent(message)

        messages = await self._build_messages(session_id, message, system_prompt)

        reply, tokens = await self._client.chat_complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        await self._persist_turn(session_id, message, reply)
        logger.info("chat session=%s tokens=%d", session_id, tokens)
        return reply, tokens, intent

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat turn.

        Yields token deltas. Persists the full reply to session memory
        once streaming completes.
        """
        messages = await self._build_messages(session_id, message, system_prompt)

        full_reply_parts: list[str] = []

        async for token in self._client.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            full_reply_parts.append(token)
            yield token

        full_reply = "".join(full_reply_parts)
        await self._persist_turn(session_id, message, full_reply)
        logger.info("chat_stream session=%s chars=%d", session_id, len(full_reply))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _build_messages(
        self,
        session_id: str,
        user_message: str,
        system_prompt: str | None,
    ) -> list[dict[str, str]]:
        """Assemble the messages array for the OpenAI API call."""
        history = await self._sessions.get_history(session_id)
        session_prompt = await self._sessions.get_system_prompt(session_id)

        system = system_prompt or session_prompt or _DEFAULT_SYSTEM_PROMPT
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": user_message})
        return messages

    async def _persist_turn(
        self, session_id: str, user_message: str, assistant_reply: str
    ) -> None:
        """Save both sides of the turn to session memory."""
        await self._sessions.append(
            session_id, ConversationMessage(role="user", content=user_message)
        )
        await self._sessions.append(
            session_id, ConversationMessage(role="assistant", content=assistant_reply)
        )

    async def _extract_intent(self, user_message: str) -> IntentExtraction | None:
        """Extract user intent using a structured LLM call."""
        prompt = (
            "You've just delivered the legally required disclosure. Listen to the lead's first response and classify intent. "
            "Bookkeeping prospects are often busy business owners — be respectful of their time.\n"
            "• PROCEED: \"Yes\", \"Sure\", \"Okay\" → set permission_to_continue\n"
            "• SELF-IDENTIFY: \"Yes this is {{lead_first_name}}\" → set identity_confirmed to true\n"
            "• BAD TIME: \"In a meeting\", \"With a client\", \"Driving\" → set bad_time_to_talk\n"
            "• WRONG PERSON: \"She's not here\", \"This is the office line\" → set wrong_person (DO NOT reveal business or financial details)\n"
            "• OPT OUT: \"Stop calling\" → set opted_out\n"
            "• HOSTILE: anger or \"I never asked for this\" → set lead_sentiment < -0.6\n"
            "• SILENCE 5s+: re-greet once, then exit. Keep follow-ups to 1 sentence. Stop talking if interrupted.\n\n"
            "Return ONLY a JSON object with the following schema:\n"
            "{\n"
            '  "permission_to_continue": boolean,\n'
            '  "identity_confirmed": boolean,\n'
            '  "bad_time_to_talk": boolean,\n'
            '  "wrong_person": boolean,\n'
            '  "opted_out": boolean,\n'
            '  "lead_sentiment": number (-1.0 to 1.0)\n'
            "}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ]
        try:
            content, _ = await self._client.chat_complete(
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            import json
            data = json.loads(content)
            return IntentExtraction(**data)
        except Exception as exc:
            logger.error("Failed to extract intent: %s", exc)
            return None
