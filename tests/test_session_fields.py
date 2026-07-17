import unittest
from unittest.mock import AsyncMock

from app.services.chat_service import ChatService


class DummySettings:
    session_max_messages = 50
    session_ttl_seconds = 300


class DummySessionStore:
    def __init__(self):
        self.create_session = AsyncMock()
        self.append = AsyncMock()
        self.get_history = AsyncMock(return_value=[])
        self.get_system_prompt = AsyncMock(return_value=None)


class SessionFieldsTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_session_passes_session_context_to_db_persistence(self):
        store = DummySessionStore()
        service = ChatService(
            openai_client=object(),
            session_store=store,
            settings=DummySettings(),
        )

        save_db = AsyncMock()
        service._save_session_db = save_db

        session_id = await service.create_session(
            system_prompt="system prompt",
            greeting_message="hello there",
            client_id="client-1",
            lead_id="lead-1",
        )

        self.assertTrue(session_id)
        save_db.assert_awaited_once_with(
            session_id,
            "client-1",
            "lead-1",
            "system prompt",
            "hello there",
        )
