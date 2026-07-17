import time
import requests
import uuid
import statistics
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

TEST_CASES = [
    # Original 10
    {"msg": "Yes, I have a few minutes.", "expected": []},
    {"msg": "What is this regarding?", "expected": ["Soumyajit", "meeting"]},
    {"msg": "I'm busy right now, call back later.", "expected": []},
    {"msg": "Who is this again?", "expected": ["Arun", "Actyvate"]},
    {"msg": "Yes, we can talk.", "expected": []},
    {"msg": "No, this is a bad time.", "expected": []},
    {"msg": "I don't have a meeting scheduled with Soumyajit.", "expected": []},
    {"msg": "Could you reschedule this for tomorrow?", "expected": []},
    {"msg": "Yeah, I'm free, what's up?", "expected": []},
    {"msg": "Please remove me from your list.", "expected": []},
    
    # 30 New cases about Actyvate
    {"msg": "What does Actyvate do?", "expected": ["sales enablement", "B2B", "SaaS"]},
    {"msg": "Are you a B2B or B2C company?", "expected": ["B2B"]},
    {"msg": "What is your main product?", "expected": ["Actyvate Sync"]},
    {"msg": "Does your product integrate with Salesforce?", "expected": ["Salesforce"]},
    {"msg": "Does it work with Hubspot?", "expected": ["Hubspot"]},
    {"msg": "Where is your company located?", "expected": ["Austin", "Texas"]},
    {"msg": "Who is the CEO of Actyvate?", "expected": ["John Doe"]},
    {"msg": "How much does it cost?", "expected": ["$99", "99"]},
    {"msg": "What is the name of your AI product?", "expected": ["Actyvate Sync"]},
    {"msg": "Where are you headquartered?", "expected": ["Austin", "Texas"]},
    {"msg": "Can it track lead sentiments?", "expected": ["sentiment"]},
    {"msg": "Does it automate scheduling?", "expected": ["schedule", "scheduling"]},
    {"msg": "How much per user per month?", "expected": ["99"]},
    {"msg": "Who am I speaking to?", "expected": ["Arun"]},
    {"msg": "Are you a human?", "expected": ["AI assistant", "AI"]},
    {"msg": "Is Actyvate an AI company?", "expected": ["AI", "sales enablement"]},
    {"msg": "Who is John Doe?", "expected": ["CEO"]},
    {"msg": "What is the starting price?", "expected": ["99", "$99"]},
    {"msg": "Do you do sales enablement?", "expected": ["Yes", "sales enablement"]},
    {"msg": "Is your software for B2B?", "expected": ["B2B"]},
    {"msg": "Can you extract intent from conversations?", "expected": ["intent"]},
    {"msg": "Do you integrate with any CRMs?", "expected": ["Salesforce", "Hubspot"]},
    {"msg": "Can you help my sales team?", "expected": ["sales", "enablement"]},
    {"msg": "Are you based in California?", "expected": ["Austin", "Texas"]},
    {"msg": "Is the CEO Soumyajit?", "expected": ["John Doe"]},
    {"msg": "Does it cost $50 a month?", "expected": ["99", "$99"]},
    {"msg": "Is the product called Actyvate Sync?", "expected": ["Actyvate Sync", "Yes"]},
    {"msg": "Do you do B2B SaaS?", "expected": ["B2B", "SaaS"]},
    {"msg": "Can I track sentiments with this?", "expected": ["sentiment"]},
    {"msg": "Tell me about your company.", "expected": ["Actyvate", "Austin", "B2B"]}
]

def create_session(client_id: str, lead_id: str) -> tuple[str, float]:
    """Create a session and return (session_id, time_taken_seconds)."""
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
    """Send a chat message and return (response_text, time_taken_seconds, tokens_used)."""
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

def run_tests() -> List[Dict[str, Any]]:
    results = []
    
    print(f"{'#':<3} | {'Test Case':<45} | {'Valid?':<6} | {'Sess(s)':<8} | {'Chat(s)':<8}")
    print("-" * 80)
    
    for i, test in enumerate(TEST_CASES, 1):
        test_msg = test["msg"]
        expected = test["expected"]
        
        client_id = f"test_client_{uuid.uuid4().hex[:8]}"
        lead_id = f"test_lead_{uuid.uuid4().hex[:8]}"
        
        try:
            session_id, session_time = create_session(client_id, lead_id)
            reply, chat_time, tokens_used = send_chat_message(session_id, test_msg)
            
            is_valid = check_validation(reply, expected)
            
            results.append({
                "id": i,
                "test_case": test_msg,
                "expected_keywords": expected,
                "session_time": session_time,
                "chat_time": chat_time,
                "tokens_used": tokens_used,
                "success": True,
                "ai_response": reply,
                "is_valid": is_valid
            })
            
            val_str = "Yes" if is_valid else "No"
            print(f"{i:<3} | {test_msg[:43]:<45} | {val_str:<6} | {session_time:<8.4f} | {chat_time:<8.4f}")
            
        except Exception as e:
            print(f"{i:<3} | {test_msg[:43]:<45} | ERR    | N/A      | N/A")
            results.append({
                "id": i,
                "test_case": test_msg,
                "expected_keywords": expected,
                "success": False,
                "error": str(e)
            })
            
    return results

def generate_csv_report(results: List[Dict[str, Any]]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_api_perf_report_{timestamp}.csv"
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Test ID", "User Message", "Success", "Session Creation Time (s)", 
            "Chat Response Time (s)", "Tokens Used", "AI Response", "Expected Keywords", "Validation Passed", "Error"
        ])
        
        for r in results:
            if r["success"]:
                writer.writerow([
                    r["id"], r["test_case"], True, f"{r['session_time']:.4f}", 
                    f"{r['chat_time']:.4f}", r["tokens_used"], r["ai_response"], 
                    ", ".join(r["expected_keywords"]), r["is_valid"], ""
                ])
            else:
                writer.writerow([
                    r["id"], r["test_case"], False, "", "", "", "", 
                    ", ".join(r["expected_keywords"]), False, r["error"]
                ])
                
    print(f"\nDetailed CSV report saved to '{filename}'")

if __name__ == "__main__":
    print("Starting Chat API Performance & Validation Tests...\n")
    results = run_tests()
    generate_csv_report(results)
