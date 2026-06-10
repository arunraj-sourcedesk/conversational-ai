import os
import sys
# Ensure repository root is on sys.path for imports when run as a script
sys.path.insert(0, os.getcwd())
from app.services.actyvate_prep_agent import PrepCallAgent


def main():
    flow_path = os.path.join(os.getcwd(), "app", "workflows", "actyvate_prep_call_flow.json")
    agent = PrepCallAgent(flow_path)
    agent.start({
        "agent_name": "PrepAgent",
        "client_name": "Virtuous Bookkeeping",
        "rep_name": "Alex",
        "meeting_datetime_relative": "tomorrow at 2pm",
        "lead_first_name": "Sam",
    })

    print("START ->", agent.current_node)

    # Simulate lead says 'Yes' to permission
    agent.transition({"permission_to_continue": True})
    print("After permission ->", agent.current_node)

    # Simulate identity confirmed
    agent.transition({"identity_confirmed": True})
    print("After identity ->", agent.current_node)

    # Simulate meeting confirmed
    agent.transition({"primary_intent": "confirm"})
    print("After intent ->", agent.current_node)


if __name__ == "__main__":
    main()
