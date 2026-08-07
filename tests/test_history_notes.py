import unittest
from unittest.mock import AsyncMock, patch

from app.models.chat import ConversationMessage
from app.services.chat_service import ChatService


class DummySettings:
    session_max_messages = 50
    session_ttl_seconds = 300


class FakeOpenAIClient:
    def __init__(self, payload):
        self.payload = payload
        self.call_count = 0

    async def chat_complete(self, messages, temperature=0.7, max_tokens=1024, response_format=None):
        self.call_count += 1
        return self.payload, 123


class SessionNotesTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_and_persist_session_notes_only_once(self):
        client = FakeOpenAIClient("Summary\n- Key point\nNext steps: follow up")
        store = type("DummyStore", (), {})()
        service = ChatService(openai_client=client, session_store=store, settings=DummySettings())

        history = [
            ConversationMessage(role="user", content="Hi, I want to learn more"),
            ConversationMessage(role="assistant", content="Happy to help"),
        ]

        with patch("app.utils.db.get_session", new=AsyncMock(return_value={"notes": None})), patch(
            "app.utils.db.save_session", new=AsyncMock()
        ) as save_session:
            notes = await service.get_session_notes("session-123", history=history)

        self.assertEqual(notes, "Summary\n- Key point\nNext steps: follow up")
        self.assertEqual(client.call_count, 1)
        save_session.assert_awaited_once()

        with patch("app.utils.db.get_session", new=AsyncMock(return_value={"notes": "Stored notes"})), patch(
            "app.utils.db.save_session", new=AsyncMock()
        ) as save_session:
            notes = await service.get_session_notes("session-123", history=history)

        self.assertEqual(notes, "Stored notes")
        self.assertEqual(client.call_count, 1)
        save_session.assert_not_awaited()

    async def test_notes_prompt_forbids_placeholders(self):
        last_messages = []

        class InspectingClient:
            async def chat_complete(self, messages, temperature=0.7, max_tokens=1024, response_format=None):
                nonlocal last_messages
                last_messages = messages
                return "Valid summary", 50

        service = ChatService(openai_client=InspectingClient(), session_store=type("DummyStore", (), {})(), settings=DummySettings())
        history = [ConversationMessage(role="user", content="Hello")]

        with patch("app.utils.db.get_session", new=AsyncMock(return_value={"notes": None})), patch(
            "app.utils.db.save_session", new=AsyncMock()
        ):
            await service.get_session_notes("session-999", history=history)

        system_msg = next(m["content"] for m in last_messages if m["role"] == "system")
        self.assertIn("Google Workspace / Gemini Notes style", system_msg)
        self.assertIn("NEVER output template placeholders", system_msg)
        self.assertIn("[Insert Date]", system_msg)


if __name__ == "__main__":
    unittest.main()
