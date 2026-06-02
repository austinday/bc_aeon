import requests
import json
import base64

# The server is running in a container, we assume port 8000 is mapped to 8000 on host
URL = "http://localhost:8000"

def test_ping():
    print("Testing /ping...")
    try:
        r = requests.get(f"{URL}/ping")
        print(f"Ping response: {r.status_code} - {r.text}")
        return r.status_code == 200
    except Exception as e:
        print(f"Ping failed: {e}")
        return False

def test_hold():
    print("\nTesting /debug_press_and_hold...")
    # We need a session first
    nav_payload = {
        "session_id": "test_session",
        "url": "data:text/html,<html><body><button id='hold_me'>Hold Me</button><div id='status'>Not Held</div><script>const b=document.getElementById('hold_me'); b.onmousedown=()=>{document.getElementById('status').innerText='Held!';}; b.onmouseup=()=>{document.getElementById('status').innerText='Not Held'};</script></body></html>"
    }
    try:
        requests.post(f"{URL}/navigate", json=nav_payload)
        
        hold_payload = {
            "session_id": "test_session",
            "action": "press_and_hold",
            "selector": "#hold_me",
            "duration": 3000
        }
        # Using the debug endpoint to bypass dispatcher
        r = requests.post(f"{URL}/debug_press_and_hold", json=hold_payload)
        print(f"Hold request response: {r.status_code} - {r.text}")
        
        # Verify result using debug_get_text
        text_payload = {
            "session_id": "test_session",
            "selector": "#status"
        }
        res_text = requests.post(f"{URL}/debug_get_text", json=text_payload)
        print(f"Status text: {res_text.json().get('text')}")
        return res_text.json().get('text') == 'Held!'
    except Exception as e:
        print(f"Hold test failed: {e}")
        return False

if __name__ == "__main__":
    if test_ping():
        if test_hold():
            print("\nSUCCESS: Press and hold triggered!")
        else:
            print("\nFAILURE: Press and hold did not trigger 'Held!' state.")
    else:
        print("\nFAILURE: Server not reachable.")