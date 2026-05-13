import requests
import json
import time

BASE_URL = "http://localhost:8000"
HTML_PATH = "file:///app/scripts/debug/browser_validation_v2.html"
SESSION_ID = "comp_test_session_456"
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

def navigate():
    payload = {"url": HTML_PATH, "session_id": SESSION_ID, "tab_id": TAB_ID}
    resp = requests.post(f"{BASE_URL}/navigate", json=payload)
    resp.raise_for_status()
    return resp.json()

def interact(action, element_id=None, text=None, expected_text=None):
    payload = {
        "action": action,
        "element_id": element_id,
        "text": text,
        "expected_text": expected_text,
        "session_id": SESSION_ID,
        "tab_id": TAB_ID
    }
    resp = requests.post(f"{BASE_URL}/interact", json=payload)
    if resp.status_code == 422:
        print(f"\nDEBUG: 422 Validation Error for action '{action}': {resp.text}")
    resp.raise_for_status()
    return resp.json()

def test_dropdown():
    state = navigate()
    dropdown = next((el for el in state['elements'] if el['tag'] == 'select'), None)
    if not dropdown:
        print(f"\nDEBUG: Elements found: {json.dumps(state['elements'], indent=2)}")
        raise Exception("Dropdown not found")
    
    res = interact("select", element_id=dropdown['id'], text="Banana")
    if "Banana" not in res['markdown']:
        raise Exception(f"Dropdown value not updated. Markdown: {res['markdown']}")
    return True

def test_form_input():
    state = navigate()
    input_field = next((el for el in state['elements'] if el['tag'] == 'input'), None)
    submit_btn = next((el for el in state['elements'] if el['tag'] == 'button' and "Submit" in el['text']), None)
    
    if not input_field:
        raise Exception("Input field not found")
    if not submit_btn:
        raise Exception(f"Submit button not found. Available elements: {state['elements']}")
    
    interact("type", element_id=input_field['id'], text="AeonBot")
    res = interact("click", element_id=submit_btn['id'])
    if "Hello, AeonBot!" not in res['markdown']:
        raise Exception(f"Form submission failed. Markdown: {res['markdown']}")
    return True

def test_popups():
    state = navigate()
    btn = next((el for el in state['elements'] if el['text'].startswith("Open 3 Popups")), None)
    if not btn: raise Exception("Popup button not found")
    
    res = interact("click", element_id=btn['id'])
    open_tabs = res.get('open_tabs', [])
    popups = [t for t in open_tabs if 'popup_' in t]
    if len(popups) < 3:
        raise Exception(f"Expected at least 3 popups, found {len(popups)}: {popups}")
    return True

def test_safety_lock():
    state = navigate()
    btn = next((el for el in state['elements'] if "Safety Lock Test Button" in el['text']), None)
    if not btn: raise Exception("Safety button not found")
    
    # This should FAIL because the expected text is wrong
    payload = {
        "action": "click",
        "element_id": btn['id'],
        "expected_text": "WRONG TEXT",
        "session_id": SESSION_ID,
        "tab_id": TAB_ID
    }
    resp = requests.post(f"{BASE_URL}/interact", json=payload)
    res = resp.json()
    if res.get('status') != 'error' or "Safety Lock Triggered" not in res.get('msg', ''):
        raise Exception(f"Safety lock did not trigger! Response: {res}")
    return True

def test_scrolling_and_som():
    state = navigate()
    if any("I am at the bottom!" in el['text'] for el in state['elements']):
        raise Exception("Bottom button should not be visible initially")
    
    interact("scroll_down")
    res = interact("scroll_down")
    
    bottom_btn = next((el for el in res['elements'] if "I am at the bottom!" in el['text']), None)
    if not bottom_btn:
        raise Exception("Bottom button not found after scrolling")
    return True

def test_offscreen_exclusion():
    state = navigate()
    if any("I am off-screen!" in el['text'] for el in state['elements']):
        raise Exception("Off-screen element should have been excluded from SOM")
    return True

if __name__ == "__main__":
    print(f"Starting Comprehensive Browser Validation against {BASE_URL}")
    print("-" * 50)
    
    tests = [
        ("Dropdown Selection", test_dropdown),
        ("Form Input & Submit", test_form_input),
        ("Multiple Popups", test_popups),
        ("Safety Lock", test_safety_lock),
        ("Scrolling & SOM", test_scrolling_and_som),
        ("Off-screen Exclusion", test_offscreen_exclusion),
    ]
    
    all_passed = True
    for name, func in tests:
        if test_step(name, func) is None:
            all_passed = False
            
    print("-" * 50)
    if all_passed:
        print("ALL COMPREHENSIVE TESTS PASSED! ✅")
    else:
        print("SOME TESTS FAILED. ❌")
        exit(1)