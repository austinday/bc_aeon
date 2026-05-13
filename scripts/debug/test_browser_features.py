import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"
# Use the path as it exists inside the Docker container
HTML_PATH = "file:///app/scripts/debug/browser_validation.html"
SESSION_ID = "test_session_123"
TAB_ID = "default"

def test_step(name, func):
    print(f"Testing {name}...", end=" ", flush=True)
    try:
        result = func()
        print("✅ SUCCESS")
        return result
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None

def test_navigate():
    payload = {"url": HTML_PATH, "session_id": SESSION_ID, "tab_id": TAB_ID}
    resp = requests.post(f"{BASE_URL}/navigate", json=payload)
    resp.raise_for_status()
    return resp.json()

def test_select_dropdown():
    # First, we need the element_id of the dropdown. 
    # We'll get it from the last state.
    state = test_navigate()
    dropdown = next((el for el in state['elements'] if el['tag'] == 'select'), None)
    if not dropdown:
        raise Exception("Dropdown element not found in page state")
    
    print(f"Found dropdown ID {dropdown['id']}. Selecting 'Option 2 (Banana)'...", end=" ", flush=True)
    payload = {
        "action": "select",
        "element_id": dropdown['id'],
        "text": "Option 2 (Banana)",
        "session_id": SESSION_ID,
        "tab_id": TAB_ID
    }
    resp = requests.post(f"{BASE_URL}/interact", json=payload)
    resp.raise_for_status()
    res_json = resp.json()
    
    if "Selected: option2" in res_json['markdown']:
        print("✅ SUCCESS")
    else:
        print("❌ FAILED: Dropdown value not updated in markdown")
    return res_json

def test_popup_handling():
    state = test_navigate()
    btn = next((el for el in state['elements'] if "Open Popup" in el['text']), None)
    if not btn:
        raise Exception("Popup button not found")
    
    print(f"Clicking popup button ID {btn['id']}...", end=" ", flush=True)
    payload = {
        "action": "click",
        "element_id": btn['id'],
        "session_id": SESSION_ID,
        "tab_id": TAB_ID
    }
    resp = requests.post(f"{BASE_URL}/interact", json=payload)
    resp.raise_for_status()
    res_json = resp.json()
    
    # Check if a new tab starting with 'popup_' was created
    open_tabs = res_json.get('open_tabs', [])
    popup_tabs = [t for t in open_tabs if 'popup_' in t]
    
    if popup_tabs:
        print(f"✅ SUCCESS (Found popups: {popup_tabs})")
    else:
        print("❌ FAILED: No popup tab detected in open_tabs")
    return res_json

if __name__ == "__main__":
    print(f"Starting Browser Validation Tests against {BASE_URL}")
    print("-" * 50)
    
    # 1. Navigate
    nav_res = test_step("Navigation", test_navigate)
    if not nav_res: exit(1)
    
    # 2. Select
    sel_res = test_step("Dropdown Selection", test_select_dropdown)
    
    # 3. Popup
    pop_res = test_step("Popup Handling", test_popup_handling)
    
    print("-" * 50)
    print("Tests Complete.")