import time
import requests
import uuid
import csv
from datetime import datetime
from typing import List, Dict, Any

BASE_URL = "http://localhost:8000"

SYSTEM_PROMPT = (
    "You are Arun, an AI assistant calling on behalf of Actyvate. "
    "Actyvate is a B2B SaaS company that provides AI-driven sales enablement software. "
    "Our platform helps sales teams automate meeting scheduling, extract intent from conversations, and track lead sentiments. "
    "We are headquartered in Austin, Texas, and our CEO is John Doe. "
    "Our main product is 'Actyvate Sync' which integrates with Salesforce and Hubspot. "
    "Pricing starts at $99/month per user."
)

GREETING_MESSAGE = (
    "Hi, this is Arun, an AI assistant calling on behalf of Actyvate. "
    "I'm reaching out about your upcoming meeting with Soumyajit on 29th May, 7.30 pm. "
    "This call may be recorded for quality. Is now a good time to chat for few minutes?"
)

CONVERSATION_TURNS = [
    {"msg": "Hello, who is this?", "expected": ["Arun", "Actyvate"]},
    {"msg": "Oh, Actyvate? What do you do?", "expected": ["sales enablement", "scheduling"]},
    {"msg": "Do you integrate with Salesforce?", "expected": ["Salesforce"]},
    {"msg": "What about HubSpot?", "expected": ["HubSpot"]},
    {"msg": "Where is your company located?", "expected": ["Austin", "Texas"]},
    {"msg": "Who is the CEO?", "expected": ["John Doe"]},
    {"msg": "How much does it cost?", "expected": ["99"]},
    {"msg": "Can it automate scheduling?", "expected": ["schedule", "scheduling"]},
    {"msg": "Can it track sentiments?", "expected": ["sentiment"]},
    {"msg": "Wait, did you say you are an AI?", "expected": ["AI", "assistant"]},
    {"msg": "Is your software for B2B or B2C?", "expected": ["B2B"]},
    {"msg": "Can you extract intent from our sales calls?", "expected": ["intent"]},
    {"msg": "Are you based in California?", "expected": ["Austin", "Texas"]},
    {"msg": "What is the name of your main product?", "expected": ["Actyvate Sync"]},
    {"msg": "Is the starting price $50?", "expected": ["99"]},
    {"msg": "Who are you calling on behalf of again?", "expected": ["Actyvate"]},
    {"msg": "Do I need to sign an annual contract?", "expected": []},
    {"msg": "Is John Doe the founder too?", "expected": ["CEO", "John Doe"]},
    {"msg": "Okay, that sounds good. Thanks.", "expected": []},
    {"msg": "Alright, please send me an email with the details.", "expected": []}
]

def create_session(client_id: str, lead_id: str) -> tuple[str, float]:
    url = f"{BASE_URL}/chat/session"
    payload = {
        "clientId": client_id,
        "leadId": lead_id,
        "greetingMessage": GREETING_MESSAGE,
        "systemPrompt": SYSTEM_PROMPT
    }
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    response.raise_for_status()
    data = response.json()
    return data["sessionId"], (end_time - start_time)

def send_chat_message(session_id: str, message: str) -> tuple[str, float, int]:
    url = f"{BASE_URL}/chat"
    payload = {
        "sessionId": session_id,
        "message": message,
        "temperature": 0.3,
        "max_tokens": 1024
    }
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    response.raise_for_status()
    data = response.json()
    return data["response"], (end_time - start_time), data.get("tokens_used", 0)

def check_validation(reply: str, expected_keywords: List[str]) -> bool:
    if not expected_keywords:
        return True
    reply_lower = reply.lower()
    return any(kw.lower() in reply_lower for kw in expected_keywords)

def run_conversation_test() -> tuple[float, List[Dict[str, Any]]]:
    results = []
    
    client_id = f"test_client_{uuid.uuid4().hex[:8]}"
    lead_id = f"test_lead_{uuid.uuid4().hex[:8]}"
    
    print("Creating Session...")
    session_id, session_time = create_session(client_id, lead_id)
    print(f"Session Created: {session_id} in {session_time:.4f}s\n")
    
    print(f"{'Turn':<4} | {'User Message':<35} | {'Valid?':<6} | {'Chat(s)':<8} | {'Tokens':<6}")
    print("-" * 70)
    
    for i, turn in enumerate(CONVERSATION_TURNS, 1):
        test_msg = turn["msg"]
        expected = turn["expected"]
        
        try:
            reply, chat_time, tokens_used = send_chat_message(session_id, test_msg)
            is_valid = check_validation(reply, expected)
            
            results.append({
                "turn": i,
                "test_case": test_msg,
                "expected_keywords": expected,
                "chat_time": chat_time,
                "tokens_used": tokens_used,
                "success": True,
                "ai_response": reply,
                "is_valid": is_valid
            })
            
            val_str = "Yes" if is_valid else "No"
            print(f"{i:<4} | {test_msg[:33]:<35} | {val_str:<6} | {chat_time:<8.4f} | {tokens_used:<6}")
            
        except Exception as e:
            print(f"{i:<4} | {test_msg[:33]:<35} | ERR    | N/A      | N/A")
            results.append({
                "turn": i,
                "test_case": test_msg,
                "expected_keywords": expected,
                "success": False,
                "error": str(e)
            })
            
    return session_time, results

def generate_csv_report(session_time: float, results: List[Dict[str, Any]]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_conversation_perf_{timestamp}.csv"
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Session Creation Time (s)", f"{session_time:.4f}"])
        writer.writerow([])
        writer.writerow([
            "Turn", "User Message", "Success", "Chat Response Time (s)", 
            "Tokens Used", "AI Response", "Expected Keywords", "Validation Passed", "Error"
        ])
        
        for r in results:
            if r["success"]:
                writer.writerow([
                    r["turn"], r["test_case"], True, f"{r['chat_time']:.4f}", 
                    r["tokens_used"], r["ai_response"], 
                    ", ".join(r["expected_keywords"]), r["is_valid"], ""
                ])
            else:
                writer.writerow([
                    r["turn"], r["test_case"], False, "", "", "", 
                    ", ".join(r["expected_keywords"]), False, r["error"]
                ])
                
    print(f"\nDetailed CSV report saved to '{filename}'")

if __name__ == "__main__":
    print("Starting Multi-turn Conversation Performance Tests...\n")
    session_time, results = run_conversation_test()
    generate_csv_report(session_time, results)
