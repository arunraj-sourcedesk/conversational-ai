"""
Dependency injection wiring.

FastAPI's `Depends()` system is used throughout routes to get
fully-configured service instances. All heavy objects (OpenAI client,
session store) are singletons created at startup and reused.
"""

from functools import lru_cache

from fastapi import Depends

from app.clients.openai_client import OpenAIClient
from app.core.config import Settings, get_settings
from app.services.chat_service import ChatService
from app.services.session_store import InMemorySessionStore


# ---------------------------------------------------------------------------
# Singletons (created once per process)
# ---------------------------------------------------------------------------

@lru_cache()
def get_openai_client() -> OpenAIClient:
    """Return the singleton OpenAI async client."""
    return OpenAIClient(get_settings())


@lru_cache()
def get_session_store() -> InMemorySessionStore:
    """Return the singleton in-memory session store."""
    return InMemorySessionStore(get_settings())


# ---------------------------------------------------------------------------
# Per-request services (stateless, cheap to construct)
# ---------------------------------------------------------------------------

def get_chat_service(
    client: OpenAIClient = Depends(get_openai_client),
    store: InMemorySessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
) -> ChatService:
    """Inject a ChatService with its dependencies resolved."""
    return ChatService(client, store, settings)
