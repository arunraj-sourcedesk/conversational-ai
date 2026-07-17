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
            '{"attendance_intent": true, "reschedule_intent": false, "cancel_intent": false, "high_priority_discovery": {"business_type_and_tenure": "SaaS company, 6 years", "current_software_and_books_status": "QuickBooks Online, books are current", "primary_pain_and_why_now": "Need cleaner reporting for growth"}, "document_readiness_confirmation": "Documents are ready", "questions_for_jordan": ["Can you walk me through implementation?"], "nice_to_have": {"transaction_volume": "100/month", "payroll": "No", "decision_makers": "Owner and controller", "timeline": "Next 2 weeks"}}'
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
        self.assertEqual(outcome.high_priority_discovery.business_type_and_tenure, "SaaS company, 6 years")
        self.assertEqual(outcome.high_priority_discovery.current_software_and_books_status, "QuickBooks Online, books are current")
        self.assertEqual(outcome.high_priority_discovery.primary_pain_and_why_now, "Need cleaner reporting for growth")
        self.assertEqual(outcome.document_readiness_confirmation, "Documents are ready")
        self.assertEqual(outcome.questions_for_jordan, ["Can you walk me through implementation?"])
        self.assertEqual(outcome.nice_to_have.transaction_volume, "100/month")
        self.assertEqual(outcome.nice_to_have.payroll, "No")
        self.assertEqual(outcome.nice_to_have.decision_makers, "Owner and controller")
        self.assertEqual(outcome.nice_to_have.timeline, "Next 2 weeks")
        self.assertIn("We run a SaaS company", self.fake_client.messages[-1]["content"])

    async def test_get_session_outcome_returns_cached_outcome_without_calling_llm(self):
        await self.store.create_session("session-456")
        cached_outcome = SessionOutcomeResponse(
            attendance_intent=False,
            reschedule_intent=True,
            cancel_intent=False,
            document_readiness_confirmation="Cached result",
            questions_for_jordan=["Need a different time"],
        )
        await self.store.set_outcome("session-456", cached_outcome)

        outcome = await self.service.get_session_outcome("session-456")

        self.assertFalse(outcome.attendance_intent)
        self.assertTrue(outcome.reschedule_intent)
        self.assertFalse(outcome.cancel_intent)
        self.assertEqual(outcome.document_readiness_confirmation, "Cached result")
        self.assertEqual(outcome.questions_for_jordan, ["Need a different time"])
        self.assertEqual(self.fake_client.call_count, 0)

    async def test_get_session_outcome_persists_outcome_to_db(self):
        await self.store.create_session("session-789")

        with patch.object(self.service, "_save_session_outcome_db", new_callable=AsyncMock) as save_outcome:
            outcome = await self.service.get_session_outcome("session-789")

        save_outcome.assert_awaited_once_with("session-789", outcome)


if __name__ == "__main__":
    unittest.main()
