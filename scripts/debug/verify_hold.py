import requests
import time
import os

# The browser server is running in a container, mapped to 8030 on host
BASE_URL = "http://localhost:8030"

def test_hold():
    print("--- Starting Hold Verification ---")
    
    # 1. Navigate to the local test file
    # Since it's on the host, we use a data URI or a local path if the browser can access it.
    # For simplicity, we'll use a data URI containing the HTML.
    with open("scripts/debug/event_logger.html", "r") as f:
        html_content = f.read()
    
    import base64
    b64_html = base64.b64encode(html_content.encode()).decode()
    url = f"data:text/html;base64,{b64_html}"

    print(f"Navigating to test page...")
    try:
        resp = requests.post(f"{BASE_URL}/navigate", json={
            "session_id": "debug_session",
            "url": f"data:text/html;base64,{b64_html}"
        })
        print(f"Navigate status: {resp.status_code}")
    except Exception as e:
        print(f"Navigation failed: {e}")
        return

    # 2. Try to trigger the hold using the debug endpoint
    # We target the 'div' which is the only element on the page
    print("Attempting debug_press_and_hold...")
    try:
        # We use a simple selector since it's the only div
        payload = {
            "session_id": "debug_session",
            "selector": "#target",
            "duration": 3000
        }
        resp = requests.post(f"{BASE_URL}/debug_press_and_hold", json=payload)
        print(f"Hold request status: {resp.status_code}, body: {resp.text}")
    except Exception as e:
        print(f"Hold request failed: {e}")
        return

    # 3. Verify the result by getting the page text/content
    print("Verifying result...")
    try:
        # Use the debug endpoint to get the log content
        payload = {
            "session_id": "debug_session",
            "selector": "#log"
        }
        resp = requests.post(f"{BASE_URL}/debug_get_text", json=payload)
        print("--- EVENT LOG ---")
        print(resp.text)
        print("--- END LOG ---")
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    test_hold()