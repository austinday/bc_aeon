import requests
import time
import os

# The browser server is running on host port 8002
BASE_URL = "http://localhost:8002"
TEST_PAGE_PATH = os.path.abspath("scripts/debug/browser_feature_test.html")
# Ensure the path is translated for the container (/home/aday/bc_aeon -> /app)
CONTAINER_PATH = TEST_PAGE_PATH.replace('/home/aday/bc_aeon', '/app')
TEST_URL = f"file://{CONTAINER_PATH}"

def test_health():
    print("Testing health endpoint...", end=" ")
    try:
        r = requests.get(f"{BASE_URL}/health")
        if r.status_code == 200:
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL ({e})")
    return False

def run_tests():
    print(f"Starting Browser Backend Verification on {BASE_URL}...")
    print(f"Target URL: {TEST_URL}")
    
    session_id = "unified_test_session"
    tab_id = "default"
    success = True

    # 1. Navigate to the test page
    print("Navigating to test page...", end=" ")
    try:
        nav_res = requests.post(f"{BASE_URL}/navigate", json={
            "url": TEST_URL,
            "session_id": session_id,
            "tab_id": tab_id
        }).json()
        
        if nav_res.get("status") == "error":
            print(f"FAIL (Navigate error: {nav_res.get('msg')})")
            return False
        print("PASS")
    except Exception as e:
        print(f"FAIL ({e})")
        return False

    # 2. Test Dropdown
    print("Testing dropdown selection...", end=" ")
    try:
        # Find the dropdown element
        dropdown_el = next((el for el in nav_res.get("elements", []) if el["tag"] == "select"), None)
        if not dropdown_el:
            print("FAIL (Dropdown element not found in DOM)")
            success = False
        else:
            el_id = dropdown_el["id"]
            interact_res = requests.post(f"{BASE_URL}/interact", json={
                "action": "select",
                "element_id": el_id,
                "text": "Banana",
                "session_id": session_id,
                "tab_id": tab_id
            }).json()
            
            if interact_res.get("status") == "error":
                print(f"FAIL (Interact error: {interact_res.get('msg')})")
                success = False
            else:
                elements = interact_res.get("elements", [])
                updated_dropdown = next((el for el in elements if el["id"] == el_id), None)
                if updated_dropdown and "Selected: Banana" in updated_dropdown["text"]:
                    print("PASS")
                else:
                    print(f"FAIL (Selection not verified. Text: {updated_dropdown['text'] if updated_dropdown else 'None'})")
                    success = False
    except Exception as e:
        print(f"FAIL ({e})")
        success = False

    # 3. Test Popup
    print("Testing popup tracking...", end=" ")
    try:
        # We use the latest state from the dropdown interaction
        current_elements = interact_res.get("elements", []) if 'interact_res' in locals() else nav_res.get("elements", [])
        
        popup_btn = next((el for el in current_elements if "OPEN_POPUP_NOW" in el["text"]), None)
        if not popup_btn:
            print("FAIL (Popup button 'OPEN_POPUP_NOW' not found)")
            # Debug info
            print(f"\nDebug - Found elements: {[el['text'] for el in current_elements]}")
            success = False
        else:
            interact_res_popup = requests.post(f"{BASE_URL}/interact", json={
                "action": "click",
                "element_id": popup_btn["id"],
                "session_id": session_id,
                "tab_id": tab_id
            }).json()
            
            if interact_res_popup.get("status") == "error":
                print(f"FAIL (Interact error: {interact_res_popup.get('msg')})")
                success = False
            else:
                open_tabs = interact_res_popup.get("open_tabs", [])
                if len(open_tabs) > 1:
                    print(f"PASS (Tabs found: {open_tabs})")
                else:
                    print(f"FAIL (No popup tracked. Tabs: {open_tabs})")
                    success = False
    except Exception as e:
        print(f"FAIL ({e})")
        success = False

    return success

if __name__ == "__main__":
    if not test_health():
        print("\nHealth check failed. Is the server running on port 8002?")
        exit(1)
        
    if run_tests():
        print("\nALL BACKEND TESTS PASSED!")
        exit(0)
    else:
        print("\nSOME TESTS FAILED.")
        exit(1)