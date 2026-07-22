import unittest

from app.models.chat import ChatHistoryMessage, ChatHistoryResponse


class HistoryResponseFieldTests(unittest.TestCase):
    def test_history_response_uses_requested_json_field_names(self):
        response = ChatHistoryResponse(
            session_id="session-123",
            system_prompt="You are helpful",
            greeting_message="Hello there",
            page=1,
            page_size=20,
            conversation=[
                ChatHistoryMessage(
                    speaker="USER",
                    text="Hello",
                    timestamp="2024-01-01T00:00:00Z",
                )
            ],
        )

        payload = response.model_dump(by_alias=True)

        self.assertIn("conversation", payload)
        self.assertNotIn("messages", payload)
        self.assertEqual(payload["conversation"][0]["speaker"], "USER")
        self.assertEqual(payload["conversation"][0]["text"], "Hello")
        self.assertNotIn("sender", payload["conversation"][0])
        self.assertNotIn("message", payload["conversation"][0])


if __name__ == "__main__":
    unittest.main()
