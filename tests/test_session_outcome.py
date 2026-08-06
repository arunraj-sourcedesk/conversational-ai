import os
import unittest
from unittest.mock import AsyncMock, patch

from app.core.config import Settings
from app.models.chat import ConversationMessage, SessionOutcomeResponse
from app.services.chat_service import ChatService
from app.services.session_store import InMemorySessionStore


class FakeOpenAIClient:
    def __init__(self, payload):
        self.payload = payload
        self.call_count = 0

    async def chat_complete(self, messages, temperature=0.7, max_tokens=1024, response_format=None):
        self.call_count += 1
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        return self.payload, 123


class SessionOutcomeServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        settings = Settings()
        self.store = InMemorySessionStore(settings)
        self.fake_client = FakeOpenAIClient(
            '{"attendance_intent": true, "reschedule_intent": false, "cancel_intent": false, "summary": "Discussed a business need and follow-up plan", "key_points": ["Client shared goals", "Needs follow-up"], "next_steps": ["Send proposal", "Book next conversation"], "context": {"topic": "business need"}}'
        )
        self.service = ChatService(self.fake_client, self.store, settings)

    async def test_get_session_outcome_uses_chat_history_and_returns_structured_summary(self):
        await self.store.create_session("session-123")
        await self.store.append("session-123", ConversationMessage(role="user", content="Yes, I can talk now."))
        await self.store.append("session-123", ConversationMessage(role="assistant", content="Great, let’s get started."))
        await self.store.append("session-123", ConversationMessage(role="user", content="We run a SaaS company and have been in business for six years."))

        outcome = await self.service.get_session_outcome("session-123")

        self.assertTrue(outcome.attendance_intent)
        self.assertFalse(outcome.reschedule_intent)
        self.assertFalse(outcome.cancel_intent)
        self.assertEqual(outcome.summary, "Discussed a business need and follow-up plan")
        self.assertEqual(outcome.key_points, ["Client shared goals", "Needs follow-up"])
        self.assertEqual(outcome.next_steps, ["Send proposal", "Book next conversation"])
        self.assertEqual(outcome.context, {"topic": "business need"})

    async def test_get_session_outcome_accepts_generic_payload(self):
        await self.store.create_session("session-456")
        await self.store.append("session-456", ConversationMessage(role="user", content="Need help with the current issue"))
        self.fake_client.payload = '{"attendance_intent": true, "reschedule_intent": false, "cancel_intent": false, "summary": "Discussed a business need and follow-up plan", "key_points": ["Client shared goals", "Needs follow-up"], "next_steps": ["Send proposal", "Book next conversation"], "context": {"topic": "business need"}}'

        outcome = await self.service.get_session_outcome("session-456")

        self.assertTrue(outcome.attendance_intent)
        self.assertEqual(outcome.summary, "Discussed a business need and follow-up plan")
        self.assertEqual(outcome.key_points, ["Client shared goals", "Needs follow-up"])
        self.assertEqual(outcome.next_steps, ["Send proposal", "Book next conversation"])
        self.assertEqual(outcome.context, {"topic": "business need"})

    async def test_get_session_outcome_returns_cached_outcome_without_calling_llm(self):
        await self.store.create_session("session-456")
        cached_outcome = SessionOutcomeResponse(
            attendance_intent=False,
            reschedule_intent=True,
            cancel_intent=False,
            summary="Cached result",
            key_points=["Need a different time"],
            next_steps=[],
            context={},
        )
        await self.store.set_outcome("session-456", cached_outcome)

        outcome = await self.service.get_session_outcome("session-456")

        self.assertFalse(outcome.attendance_intent)
        self.assertTrue(outcome.reschedule_intent)
        self.assertFalse(outcome.cancel_intent)
        self.assertEqual(outcome.summary, "Cached result")
        self.assertEqual(outcome.key_points, ["Need a different time"])
        self.assertEqual(self.fake_client.call_count, 0)

    async def test_get_session_outcome_uses_db_persisted_outcome_without_calling_llm(self):
        await self.store.create_session("session-789")
        stored_outcome = {
            "attendance_intent": True,
            "reschedule_intent": False,
            "cancel_intent": False,
            "summary": "Stored from DB",
            "key_points": ["Already saved"],
            "next_steps": ["Follow up"],
            "context": {"source": "db"},
        }

        with patch("app.utils.db.get_session", new=AsyncMock(return_value={"outcome_data": stored_outcome})):
            outcome = await self.service.get_session_outcome("session-789")

        self.assertEqual(outcome.summary, "Stored from DB")
        self.assertEqual(outcome.key_points, ["Already saved"])
        self.assertEqual(self.fake_client.call_count, 0)

    async def test_get_session_outcome_persists_outcome_to_db(self):
        await self.store.create_session("session-999")

        with patch.object(self.service, "_save_session_outcome_db", new_callable=AsyncMock) as save_outcome:
            outcome = await self.service.get_session_outcome("session-999")

        save_outcome.assert_awaited_once_with("session-999", outcome)

    async def test_get_session_outcome_logs_generated_outcome(self):
        await self.store.create_session("session-1000")
        await self.store.append("session-1000", ConversationMessage(role="user", content="Yes, I can talk now."))
        await self.store.append("session-1000", ConversationMessage(role="assistant", content="Great, let’s get started."))

        with patch.object(self.service, "_save_session_outcome_db", new_callable=AsyncMock) as save_outcome, \
                patch("app.services.chat_service.logger.info") as logger_info:
            outcome = await self.service.get_session_outcome("session-1000")

        save_outcome.assert_awaited_once_with("session-1000", outcome)
        logger_info.assert_any_call(
            "session outcome generated session=%s outcome=%s",
            "session-1000",
            outcome.model_dump(mode="json"),
        )


if __name__ == "__main__":
    unittest.main()
