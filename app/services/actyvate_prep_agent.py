import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PrepCallAgent:
    def __init__(self, flow_path: str):
        self.flow_path = flow_path
        with open(flow_path, "r") as f:
            self.flow = json.load(f)
        self.nodes = self.flow.get("nodes", {})
        self.current_node = None
        self.variables: Dict[str, Any] = {}

    def start(self, initial_variables: Optional[Dict[str, Any]] = None) -> str:
        self.variables = initial_variables.copy() if initial_variables else {}
        self.current_node = "start_disclosure"
        logger.info("Starting flow at node: %s", self.current_node)
        return self.current_node

    def get_current_node(self) -> Dict[str, Any]:
        return self.nodes.get(self.current_node, {})

    def evaluate_condition(self, condition: str) -> bool:
        if condition is None:
            return False
        cond = condition.strip()
        if cond.lower() == "true":
            return True
        # replace simple true/false tokens to Python
        cond = cond.replace("==", "==")
        cond = cond.replace(" true", " True").replace(" true", " True")
        cond = cond.replace(" false", " False").replace(" false", " False")
        # Make variables available for eval
        try:
            # safe-ish eval: allow access only to variables
            result = bool(eval(cond, {}, self.variables))
            return result
        except Exception:
            logger.debug("Failed to eval condition '%s' with vars %s", cond, self.variables)
            return False

    def select_next_node(self, node: Dict[str, Any]) -> Optional[str]:
        edges = node.get("edges", [])
        for edge in edges:
            cond = edge.get("condition", "true")
            if self.evaluate_condition(cond):
                return edge.get("next")
        return None

    def transition(self, updates: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if updates:
            # pydantic models might be passed; convert if needed
            try:
                # if pydantic BaseModel
                if hasattr(updates, "dict") and callable(getattr(updates, "dict")):
                    updates = updates.dict()
            except Exception:
                pass

            self.variables.update(updates)
            logger.info("Variables updated: %s", updates)

        node = self.get_current_node()
        if not node:
            logger.warning("No current node to transition from")
            return None

        # auto-invoke tool actions for tool nodes (by title prefix)
        title = node.get("title", "")
        if title.startswith("Tool:"):
            tool_name = title[5:].strip()
            self.invoke_tool(tool_name)

        next_node = self.select_next_node(node)
        if next_node:
            logger.info("Transitioning %s -> %s", self.current_node, next_node)
            self.current_node = next_node
            return self.current_node

        logger.info("No outgoing edge matched for node %s", self.current_node)
        self.current_node = None
        return None

    def invoke_tool(self, tool_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Router for tool stubs
        logger.info("Invoking tool: %s payload=%s", tool_name, payload)
        name = tool_name.lower()
        if "markconfirmed" in name or "mark confirmed" in name:
            return self.tool_mark_confirmed(payload)
        if "markdnc" in name or "mark dnc" in name:
            return self.tool_mark_dnc(payload)
        if "sendreschedulelink" in name or "reschedule" in name:
            return self.tool_send_reschedule_link(payload)
        if "schedulecallback" in name or "schedule callback" in name:
            return self.tool_schedule_callback(payload)
        if "fetchslots" in name or "fetch slots" in name:
            return self.tool_fetch_slots(payload)
        if "cancelmeeting" in name or "cancel meeting" in name:
            return self.tool_cancel_meeting(payload)
        if "schedule_followup_reminder" in name or "schedule followup reminder" in name:
            return self.tool_schedule_followup_reminder(payload)

        logger.warning("Unknown tool requested: %s", tool_name)
        return {"ok": False, "error": "unknown_tool"}

    # Tool stubs -------------------------------------------------
    def tool_mark_dnc(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Tool_MarkDNC called with payload: %s", payload)
        # TODO: integrate with CRM / backend
        return {"ok": True}

    def tool_mark_confirmed(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Tool_MarkConfirmed called with payload: %s", payload)
        return {"ok": True}

    def tool_send_reschedule_link(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Tool_SendRescheduleLink called with payload: %s", payload)
        return {"ok": True, "channels_sent": ["email", "whatsapp"]}

    def tool_schedule_callback(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Tool_ScheduleCallback called with payload: %s", payload)
        return {"ok": True, "scheduled_at": payload.get("preferred_callback_time") if payload else None}

    def tool_fetch_slots(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Tool_FetchSlots called with payload: %s", payload)
        # Return example slots
        return {"ok": True, "slots": ["Tomorrow 9am", "Tomorrow 2pm", "Friday 11am"]}

    def tool_cancel_meeting(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Tool_CancelMeeting called with payload: %s", payload)
        return {"ok": True}

    def tool_schedule_followup_reminder(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("Tool_ScheduleFollowupReminder called with payload: %s", payload)
        return {"ok": True}


# Note: this module provides an in-process adapter usable by ChatService. The
# ChatService will detect sessions configured for the Actyvate flow and route
# the first turn through this flow agent rather than performing a normal LLM
# completion.
