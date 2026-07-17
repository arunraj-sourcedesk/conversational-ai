"""
In-memory conversation session store.

Provides a simple async-safe session registry backed by a dict.
In production, swap this for Redis / DynamoDB by implementing the
same SessionStore interface.

Key design decisions:
- asyncio.Lock per session for concurrency safety
- TTL enforcement on every read/write
- Circular buffer: oldest messages are pruned beyond `max_messages`
"""

import asyncio
import time
from collections import deque
from typing import Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.chat import ConversationMessage, SessionOutcomeResponse

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol (interface) — swap the implementation without touching callers
# ---------------------------------------------------------------------------

class SessionStore(Protocol):
    """Abstract interface for session storage backends."""

    async def get_history(self, session_id: str) -> list[ConversationMessage]:
        """Return the ordered conversation history for a session."""
        ...

    async def append(self, session_id: str, message: ConversationMessage) -> None:
        """Append a single message to a session's history."""
        ...

    async def clear(self, session_id: str) -> None:
        """Delete all history for a session."""
        ...

    async def create_session(self, session_id: str, system_prompt: str | None = None) -> None:
        """Create a new session with an optional system prompt."""
        ...

    async def get_system_prompt(self, session_id: str) -> str | None:
        """Get the configured system prompt for a session, if any."""
        ...

    async def get_outcome(self, session_id: str) -> SessionOutcomeResponse | None:
        """Return a cached outcome for the session, if one exists."""
        ...

    async def set_outcome(self, session_id: str, outcome: SessionOutcomeResponse) -> None:
        """Cache an outcome for the session."""
        ...



# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class _SessionState:
    """Internal state container for one session."""

    def __init__(self, max_messages: int) -> None:
        self.messages: deque[ConversationMessage] = deque(maxlen=max_messages)
        self.last_active: float = time.monotonic()
        self.lock: asyncio.Lock = asyncio.Lock()
        self.system_prompt: str | None = None
        self.outcome: SessionOutcomeResponse | None = None


class InMemorySessionStore:
    """Thread-safe in-memory session store with TTL expiry."""

    def __init__(self, settings: Settings) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._max_messages = settings.session_max_messages
        self._ttl = settings.session_ttl_seconds
        self._global_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def get_history(self, session_id: str) -> list[ConversationMessage]:
        """Return conversation history, or [] for unknown / expired sessions."""
        state = await self._get_state(session_id)
        if state is None:
            return []
        async with state.lock:
            self._touch(state)
            return list(state.messages)

    async def append(self, session_id: str, message: ConversationMessage) -> None:
        """Append a message, creating the session on first use."""
        state = await self._get_or_create_state(session_id)
        async with state.lock:
            state.messages.append(message)
            self._touch(state)
        logger.debug(
            "session=%s role=%s len=%d", session_id, message.role, len(state.messages)
        )

    async def clear(self, session_id: str) -> None:
        """Remove a session entirely."""
        async with self._global_lock:
            self._sessions.pop(session_id, None)

    async def create_session(self, session_id: str, system_prompt: str | None = None) -> None:
        """Create a new session explicitly."""
        async with self._global_lock:
            state = _SessionState(self._max_messages)
            state.system_prompt = system_prompt
            self._sessions[session_id] = state
            logger.debug("session=%s created explicitly", session_id)

    async def get_system_prompt(self, session_id: str) -> str | None:
        """Return the configured system prompt, if any."""
        state = await self._get_state(session_id)
        if state is None:
            return None
        async with state.lock:
            self._touch(state)
            return state.system_prompt

    async def get_outcome(self, session_id: str) -> SessionOutcomeResponse | None:
        """Return a cached outcome for the session, if one exists."""
        state = await self._get_state(session_id)
        if state is None:
            return None
        async with state.lock:
            self._touch(state)
            return state.outcome.model_copy(deep=True) if state.outcome is not None else None

    async def set_outcome(self, session_id: str, outcome: SessionOutcomeResponse) -> None:
        """Cache an outcome for the session."""
        state = await self._get_or_create_state(session_id)
        async with state.lock:
            state.outcome = outcome.model_copy(deep=True)
            self._touch(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_state(self, session_id: str) -> _SessionState | None:
        async with self._global_lock:
            state = self._sessions.get(session_id)
            if state and self._is_expired(state):
                del self._sessions[session_id]
                logger.debug("session=%s expired, evicted", session_id)
                return None
            return state

    async def _get_or_create_state(self, session_id: str) -> _SessionState:
        async with self._global_lock:
            state = self._sessions.get(session_id)
            if state is None or self._is_expired(state):
                state = _SessionState(self._max_messages)
                self._sessions[session_id] = state
                logger.debug("session=%s created", session_id)
            return state

    def _is_expired(self, state: _SessionState) -> bool:
        return (time.monotonic() - state.last_active) > self._ttl

    @staticmethod
    def _touch(state: _SessionState) -> None:
        state.last_active = time.monotonic()
