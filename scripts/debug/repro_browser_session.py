import asyncio
import requests
import json

BASE_URL = "http://localhost:8030"

async def test_session_persistence():
    session_id = "test_session_123"
    tab_quora = "quora_tab"
    tab_gmail = "gmail_tab"
    
    # We'll use a simple page that allows us to type and check state.
    # Since we don't have a dedicated test page, we'll use a site with a search box or similar.
    # Or better, we can use a data URI to create a simple form.
    test_url = "data:text/html,<html><body><input id='test-input' type='text' value='initial'></body></html>"
    
    print("Step 1: Navigate to test page in tab 1 and type something...")
    # In a real scenario, the agent would use /interact to type.
    # We need to find the element ID first.
    res = requests.post(f"{BASE_URL}/navigate", json={
        "url": test_url,
        "session_id": session_id,
        "tab_id": tab_quora
    }).json()
    
    if res.get("status") == "error":
        print(f"Error navigating: {res}")
        return

    # Find the input element ID
    element_id = None
    for el in res.get("elements", []):
        if el.get("tag") == "input":
            element_id = el.get("id")
            break
    
    if element_id is None:
        print("Could not find input element")
        return

    print(f"Found input element ID: {element_id}. Typing 'Hello World'...")
    requests.post(f"{BASE_URL}/interact", json={
        "action": "type",
        "element_id": element_id,
        "text": "Hello World",
        "session_id": session_id,
        "tab_id": tab_quora
    })

    print("Step 2: Navigate to another page in tab 2...")
    requests.post(f"{BASE_URL}/navigate", json={
        "url": "https://www.google.com",
        "session_id": session_id,
        "tab_id": tab_gmail
    })

    print("Step 3: 'Navigate' back to tab 1 using /navigate (simulating agent behavior)...")
    # This should reload the page and lose the 'Hello World' text
    res_nav = requests.post(f"{BASE_URL}/navigate", json={
        "url": test_url,
        "session_id": session_id,
        "tab_id": tab_quora
    }).json()
    
    # Check if the input still has 'Hello World'
    found_text = False
    for el in res_nav.get("elements", []):
        if el.get("id") == element_id and "Hello World" in el.get("text", ""):
            found_text = True
    
    print(f"After /navigate, text 'Hello World' preserved: {found_text}")

    print("Step 4: 'Switch' back to tab 1 using /interact (simulating correct behavior)...")
    # First, let's reset the state by typing again
    requests.post(f"{BASE_URL}/interact", json={
        "action": "type",
        "element_id": element_id,
        "text": "Preserve Me",
        "session_id": session_id,
        "tab_id": tab_quora
    })
    
    # Switch away
    requests.post(f"{BASE_URL}/navigate", json={
        "url": "https://www.google.com",
        "session_id": session_id,
        "tab_id": tab_gmail
    })
    
    # Switch back using /interact (scroll_down is a safe no-op that brings page to front)
    res_int = requests.post(f"{BASE_URL}/interact", json={
        "action": "scroll_down",
        "session_id": session_id,
        "tab_id": tab_quora
    }).json()
    
    found_text_int = False
    for el in res_int.get("elements", []):
        if el.get("id") == element_id and "Preserve Me" in el.get("text", ""):
            found_text_int = True
            
    print(f"After /interact, text 'Preserve Me' preserved: {found_text_int}")

if __name__ == "__main__":
    # The browser service must be running. 
    # We assume it's running on localhost:8000 as per server.py
    try:
        asyncio.run(test_session_persistence())
    except Exception as e:
        print(f"Execution failed: {e}")
        print("Make sure the browser service is running (e.g., via docker or python server.py)")