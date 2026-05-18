import requests
import time
import os

# Configuration
SERVER_URL = "http://localhost:8000"
TEST_PAGE_PATH = "file:///app/scripts/debug/browser_test_page.html"
SESSION_ID = "test_session_123"
TAB_ID = "default"

def test_navigate():
    print("Testing navigation...")
    payload = {"url": TEST_PAGE_PATH, "session_id": SESSION_ID, "tab_id": TAB_ID}
    resp = requests.post(f"{SERVER_URL}/navigate", json=payload).json()
    if resp.get("status") == "success":
        print("Successfully navigated to test page.")
        return True
    print(f"Navigation failed: {resp}")
    return False

def test_dropdown():
    print("\nTesting dropdown selection...")
    # First, find the element ID for the select box
    nav_resp = requests.post(f"{SERVER_URL}/navigate", json={"url": TEST_PAGE_PATH, "session_id": SESSION_ID, "tab_id": TAB_ID}).json()
    elements = nav_resp.get("elements", [])
    select_id = next((el["id"] for el in elements if el["tag"] == "select"), None)
    
    if select_id is None:
        print("Could not find select element.")
        return False
    
    print(f"Found select element with ID: {select_id}. Selecting 'Banana'...")
    payload = {
        "action": "select",
        "element_id": select_id,
        "text": "Banana",
        "session_id": SESSION_ID,
        "tab_id": TAB_ID
    }
    resp = requests.post(f"{SERVER_URL}/interact", json=payload).json()
    if resp.get("status") == "success":
        print("Successfully performed select action.")
        return True
    print(f"Select action failed: {resp}")
    return False

def test_popup():
    print("\nTesting popup tracking...")
    # Find the popup button
    nav_resp = requests.post(f"{SERVER_URL}/navigate", json={"url": TEST_PAGE_PATH, "session_id": SESSION_ID, "tab_id": TAB_ID}).json()
    elements = nav_resp.get("elements", [])
    btn_id = next((el["id"] for el in elements if "Open Popup" in el["text"]), None)
    
    if btn_id is None:
        print("Could not find popup button.")
        return False
    
    print(f"Found popup button with ID: {btn_id}. Clicking...")
    payload = {
        "action": "click",
        "element_id": btn_id,
        "session_id": SESSION_ID,
        "tab_id": TAB_ID
    }
    requests.post(f"{SERVER_URL}/interact", json=payload)
    
    # Wait for popup to be created and tracked
    time.sleep(3)
    
    # Check open tabs
    nav_resp = requests.post(f"{SERVER_URL}/navigate", json={"url": TEST_PAGE_PATH, "session_id": SESSION_ID, "tab_id": TAB_ID}).json()
    open_tabs = nav_resp.get("open_tabs", [])
    print(f"Open tabs: {open_tabs}")
    
    if len(open_tabs) > 1:
        print("Popup successfully tracked!")
        # Try to switch to the popup
        popup_tab_id = [t for t in open_tabs if t != TAB_ID][0]
        print(f"Switching to popup tab: {popup_tab_id}...")
        switch_payload = {"session_id": SESSION_ID, "tab_id": popup_tab_id}
        switch_resp = requests.post(f"{SERVER_URL}/switch_tab", json=switch_payload).json()
        if switch_resp.get("status") == "success":
            print("Successfully switched to popup tab.")
            return True
        print(f"Switch to popup failed: {switch_resp}")
    else:
        print("Popup was not tracked.")
    return False

if __name__ == "__main__":
    if not test_navigate():
        exit(1)
    
    dropdown_ok = test_dropdown()
    popup_ok = test_popup()
    
    if dropdown_ok and popup_ok:
        print("\nALL TESTS PASSED!")
        exit(0)
    else:
        print("\nSOME TESTS FAILED.")
        exit(1)