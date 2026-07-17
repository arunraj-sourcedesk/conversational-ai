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

from app.clients.openai_client import OpenAIClient
from app.core.config import Settings
from app.core.logging import get_logger
from app.models.chat import (
    ConversationMessage,
    HighPriorityDiscovery,
    IntentExtraction,
    NiceToHaveDetails,
    SessionOutcomeResponse,
)
from app.services.session_store import InMemorySessionStore

logger = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, persuasive, and friendly human sales and support representative. "
    "You must never disclose that you are an AI, a program, or a bot. Always maintain a natural, conversational human persona. "
    "Respond in plain text unless the user explicitly asks for formatting."
    "The current lead has a meeting scheduled with you to talk about services or product. You are here to guide him through the call and ensure that the meeting is fruitful."
    "The meeting scheduled at 7 pm today, remind the lead about this and ask if he can talk now or later"
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
        self,
        system_prompt: str | None = None,
        greeting_message: str | None = None,
        client_id: str | None = None,
        lead_id: str | None = None,
    ) -> str:
        """Create a new session and return the generated session ID."""
        import uuid
        session_id = str(uuid.uuid4())
        await self._sessions.create_session(session_id, system_prompt)
        
        if greeting_message:
            await self._sessions.append(
                session_id, ConversationMessage(role="assistant", content=greeting_message)
            )

        # Persist the session metadata and optional greeting message before returning.
        await self._save_session_db(
            session_id,
            client_id,
            lead_id,
            system_prompt,
            greeting_message,
        )

        logger.info("create_session session=%s", session_id)
        return session_id

    async def _save_session_db(
        self,
        session_id: str,
        client_id: str | None,
        lead_id: str | None,
        system_prompt: str | None,
        greeting_message: str | None,
    ) -> None:
        try:
            from app.utils.db import save_session, save_message
            await save_session(
                session_id,
                client_id,
                lead_id,
                system_prompt=system_prompt,
                greeting_message=greeting_message,
            )
            if greeting_message:
                await save_message(session_id, "assistant", greeting_message)
        except Exception as e:
            logger.error("Failed to save session metadata to DB for session %s: %s", session_id, e)

    async def _save_session_outcome_db(self, session_id: str, outcome: SessionOutcomeResponse) -> None:
        try:
            from app.utils.db import save_session
            await save_session(
                session_id,
                None,
                None,
                outcome_data=outcome.model_dump(mode="json"),
            )
        except Exception as e:
            logger.error("Failed to save session outcome to DB for session %s: %s", session_id, e)

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
        session_prompt = await self._sessions.get_system_prompt(session_id)

        # If this session is configured to use the Actyvate prep-call flow,
        # route the first user turn through the local state-machine adapter
        # rather than performing a standard LLM completion.
        if session_prompt and "actyvate_prep_call" in session_prompt.lower():
            try:
                from app.services.actyvate_prep_agent import PrepCallAgent
                import os
                flow_path = os.path.join(os.getcwd(), "app", "workflows", "actyvate_prep_call_flow.json")
                agent = PrepCallAgent(flow_path)

                # On the first turn extract structured intent to populate variables
                if is_first_turn:
                    intent = await self._extract_intent(message)
                agent.start({})
                if intent:
                    agent.transition(intent)
                else:
                    # If no intent extracted, pass the raw message as a last resort
                    agent.transition({"raw_user_message": message})

                node = agent.get_current_node() or {}
                reply_text = node.get("first_message") or node.get("prompt") or "Okay."

                await self._persist_turn(session_id, message, reply_text, tokens_used=0, intent=intent)
                logger.info("chat (flow) session=%s node=%s", session_id, agent.current_node)
                return reply_text, 0, intent
            except Exception as exc:
                logger.exception("Actyvate flow adapter failed: %s", exc)
                # Fall back to regular LLM chat below

        messages = await self._build_messages(session_id, message, system_prompt)

        reply, tokens = await self._client.chat_complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # For the first turn, persist the DB row and then extract intent in a
        # single chained background task to avoid the race condition where the
        # UPDATE from intent extraction races ahead of the INSERT from persist.
        if is_first_turn:
            import asyncio as _asyncio
            _asyncio.create_task(
                self._persist_and_extract_intent(session_id, message, reply, tokens)
            )
        else:
            await self._persist_turn(session_id, message, reply, tokens_used=tokens)
        logger.info("chat session=%s tokens=%d", session_id, tokens)

        return reply, tokens, None

    async def get_session_outcome(self, session_id: str) -> SessionOutcomeResponse:
        """Summarize a session using the conversation history and an LLM."""
        cached_outcome = await self._sessions.get_outcome(session_id)
        if cached_outcome is not None:
            return cached_outcome

        history = await self._sessions.get_history(session_id)

        if not history:
            try:
                from app.utils.db import get_chat_history

                rows = await get_chat_history(session_id, page=1, page_size=200)
                history = [
                    ConversationMessage(role=item["role"], content=item["message"])
                    for item in rows
                ]
            except Exception as exc:
                logger.warning("No session history available for %s: %s", session_id, exc)

        if not history:
            outcome = SessionOutcomeResponse(
                attendance_intent=False,
                reschedule_intent=False,
                cancel_intent=False,
                high_priority_discovery=HighPriorityDiscovery(),
                document_readiness_confirmation="No conversation history was available.",
                questions_for_jordan=[],
                nice_to_have=None,
            )
            await self._sessions.set_outcome(session_id, outcome)
            await self._save_session_outcome_db(session_id, outcome)
            return outcome

        conversation_text = "\n".join(
            f"{'USER' if message.role == 'user' else 'AI'}: {message.content}"
            for message in history
        )

        prompt = (
            "You are summarizing a sales discovery call. Follow this priority order exactly:\n"
            "1. Decide whether the lead intends to attend, reschedule, or cancel. Return boolean flags for attendance_intent, reschedule_intent, and cancel_intent.\n"
            "2. Capture the three high-priority discovery answers in this order: business type & tenure; current software & books status; primary pain / why now.\n"
            "3. Confirm whether document readiness is confirmed or not.\n"
            "4. Capture any questions the lead wants Jordan to address.\n"
            "5. If present, include optional nice-to-have details: transaction volume, payroll, decision-makers, timeline.\n"
            "Tie-breaker: if the lead seems happy and wants to end warmly, prioritize a warm close and leave missing items blank or empty rather than forcing weak guesses.\n"
            "Return ONLY a JSON object with the schema: {"
            '"attendance_intent": false, '
            '"reschedule_intent": false, '
            '"cancel_intent": false, '
            '"high_priority_discovery": {"business_type_and_tenure": "", "current_software_and_books_status": "", "primary_pain_and_why_now": ""}, '
            '"document_readiness_confirmation": "", '
            '"questions_for_jordan": [], '
            '"nice_to_have": {"transaction_volume": "", "payroll": "", "decision_makers": "", "timeline": ""}}'
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Conversation history:\n{conversation_text}"},
        ]

        try:
            content, _ = await self._client.chat_complete(
                messages=messages,
                temperature=0.1,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            import json

            payload = json.loads(content)
            normalized_payload = self._normalize_session_outcome_payload(payload)
            outcome = SessionOutcomeResponse.model_validate(normalized_payload)
            await self._sessions.set_outcome(session_id, outcome)
            await self._save_session_outcome_db(session_id, outcome)
            return outcome
        except Exception as exc:
            logger.error("Failed to summarize session outcome for %s: %s", session_id, exc)
            outcome = SessionOutcomeResponse(
                attendance_intent=False,
                reschedule_intent=False,
                cancel_intent=False,
                high_priority_discovery=HighPriorityDiscovery(),
                document_readiness_confirmation="",
                questions_for_jordan=[],
                nice_to_have=None,
            )
            await self._sessions.set_outcome(session_id, outcome)
            await self._save_session_outcome_db(session_id, outcome)
            return outcome

    def _normalize_session_outcome_payload(self, payload: dict) -> dict:
        """Normalize the LLM payload into a safe response structure."""
        if any(key in payload for key in {"attendance_intent", "reschedule_intent", "cancel_intent"}):
            attendance_intent = self._coerce_bool(payload.get("attendance_intent"))
            reschedule_intent = self._coerce_bool(payload.get("reschedule_intent"))
            cancel_intent = self._coerce_bool(payload.get("cancel_intent"))
        else:
            meeting_attendance = str(payload.get("meeting_attendance", "")).strip().lower()
            if meeting_attendance in {"yes"}:
                attendance_intent, reschedule_intent, cancel_intent = True, False, False
            elif meeting_attendance in {"reschedule"}:
                attendance_intent, reschedule_intent, cancel_intent = False, True, False
            elif meeting_attendance in {"no", "cancel", "declin", "decline", "cancelled", "declined"}:
                attendance_intent, reschedule_intent, cancel_intent = False, False, True
            else:
                attendance_intent, reschedule_intent, cancel_intent = False, False, False

        discovery_payload = payload.get("high_priority_discovery") or {}
        high_priority_discovery = HighPriorityDiscovery(
            business_type_and_tenure=str(discovery_payload.get("business_type_and_tenure", "")).strip(),
            current_software_and_books_status=str(discovery_payload.get("current_software_and_books_status", "")).strip(),
            primary_pain_and_why_now=str(discovery_payload.get("primary_pain_and_why_now", "")).strip(),
        )

        nice_to_have_payload = payload.get("nice_to_have") or {}
        nice_to_have = NiceToHaveDetails(
            transaction_volume=str(nice_to_have_payload.get("transaction_volume", "") or "").strip() or None,
            payroll=str(nice_to_have_payload.get("payroll", "") or "").strip() or None,
            decision_makers=str(nice_to_have_payload.get("decision_makers", "") or "").strip() or None,
            timeline=str(nice_to_have_payload.get("timeline", "") or "").strip() or None,
        )
        if nice_to_have.transaction_volume is None and nice_to_have.payroll is None and nice_to_have.decision_makers is None and nice_to_have.timeline is None:
            nice_to_have = None

        questions = payload.get("questions_for_jordan") or []
        if isinstance(questions, str):
            questions = [q.strip() for q in questions.split(";") if q.strip()]
        elif not isinstance(questions, list):
            questions = []

        return {
            "attendance_intent": attendance_intent,
            "reschedule_intent": reschedule_intent,
            "cancel_intent": cancel_intent,
            "high_priority_discovery": high_priority_discovery.model_dump(),
            "document_readiness_confirmation": str(payload.get("document_readiness_confirmation", "")).strip(),
            "questions_for_jordan": questions,
            "nice_to_have": nice_to_have.model_dump() if nice_to_have is not None else None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_bool(self, value: object) -> bool:
        """Coerce common boolean-like values into a Python bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1"}:
                return True
            if normalized in {"false", "no", "n", "0", ""}:
                return False
        return bool(value)

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
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        persist_db: bool = True,
        tokens_used: int | None = None,
        intent: IntentExtraction | None = None,
    ) -> None:
        """Save both sides of the turn to session memory and (optionally) the DB."""
        await self._sessions.append(
            session_id, ConversationMessage(role="user", content=user_message)
        )
        await self._sessions.append(
            session_id, ConversationMessage(role="assistant", content=assistant_reply)
        )

        if persist_db:
            import asyncio
            asyncio.create_task(
                self._persist_turn_db(
                    session_id,
                    user_message,
                    assistant_reply,
                    tokens_used,
                    intent,
                )
            )

    async def _persist_turn_db(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        tokens_used: int | None = None,
        intent: IntentExtraction | None = None,
    ) -> None:
        """Persist both sides of a turn to the database."""
        intent_payload = intent.model_dump() if intent is not None else None
        try:
            from app.utils.db import save_message
            await save_message(session_id, "user", user_message)
            await save_message(
                session_id,
                "assistant",
                assistant_reply,
                tokens_used=tokens_used,
                intent_extraction=intent_payload,
            )
        except Exception as e:
            logger.error("Failed to save turn to DB for session %s: %s", session_id, e)

    async def _persist_and_extract_intent(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        tokens: int,
    ) -> None:
        """
        Background task (first turn only): persist the chat turn to the DB
        *then* extract intent and patch the assistant row.

        Sequencing both operations here eliminates the race condition where
        the UPDATE from intent extraction used to run before the INSERT from
        _persist_turn_db had committed.
        """
        # Step 1: persist the turn (INSERT) — must complete before we UPDATE.
        await self._persist_turn_db(session_id, user_message, assistant_reply, tokens)
        # Also update in-memory session store (persist_db=False since we just did it).
        await self._sessions.append(
            session_id, ConversationMessage(role="user", content=user_message)
        )
        await self._sessions.append(
            session_id, ConversationMessage(role="assistant", content=assistant_reply)
        )

        # Step 2: extract intent and UPDATE the row we just inserted.
        intent = await self._extract_intent(user_message)
        if intent is None:
            return
        try:
            from app.utils.db import update_last_assistant_intent
            await update_last_assistant_intent(
                session_id,
                intent_extraction=intent.model_dump(),
            )
            logger.info("intent updated in DB for session=%s", session_id)
        except Exception as e:
            logger.error("Failed to update intent in DB for session %s: %s", session_id, e)

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
