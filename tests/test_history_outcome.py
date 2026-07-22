import unittest
from unittest.mock import AsyncMock, patch

from app.models.chat import ChatHistoryResponse, SessionOutcomeResponse


class HistoryOutcomeTests(unittest.TestCase):
    def test_history_response_accepts_outcome_field(self):
        outcome = SessionOutcomeResponse(
            attendance_intent=True,
            reschedule_intent=False,
            cancel_intent=False,
            document_readiness_confirmation="Ready",
            questions_for_rep=["Need a follow-up"],
        )
        response = ChatHistoryResponse(
            session_id="session-1",
            system_prompt="prompt",
            greeting_message="hello",
            page=1,
            page_size=20,
            conversation=[],
            notes="Some notes",
            outcome=SessionOutcomeResponse(
                attendance_intent=True,
                summary="Generic summary",
                key_points=["Main point"],
                next_steps=["Need a follow-up"],
                context={"topic": "support"},
            ),
        )

        payload = response.model_dump(by_alias=True)
        self.assertEqual(payload["outcome"]["attendance_intent"], True)
        self.assertEqual(payload["outcome"]["next_steps"], ["Need a follow-up"])


if __name__ == "__main__":
    unittest.main()
