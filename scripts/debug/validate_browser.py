import requests
import base64
import os
import time
from pathlib import Path

API_URL = "http://localhost:8030"
OUTPUT_DIR = Path("aeon_output/browser_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_screenshot(data, filename):
    with open(OUTPUT_DIR / filename, "wb") as f:
        f.write(base64.b64decode(data))
    print(f"Saved screenshot: {filename}")

def test_step(name, payload, endpoint="/interact"):
    print(f"Testing {name}...", end=" ", flush=True)
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
        res_json = response.json()
        if res_json.get("status") == "success":
            save_screenshot(res_json["overlay_b64"], f"{name}_overlay.jpg")
            save_screenshot(res_json["clean_b64"], f"{name}_clean.jpg")
            print("SUCCESS")
            return res_json
        else:
            print(f"FAILED: {res_json.get('msg')}")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    session_id = "val_session_123"
    
    # 1. Navigate
    print("\n--- Step 1: Navigation ---")
    nav_res = test_step("navigate", {
        "url": "https://www.wikipedia.org",
        "session_id": session_id,
        "tab_id": "tab1"
    }, endpoint="/navigate")
    
    if not nav_res:
        print("Navigation failed. Aborting.")
        return

    # 2. Click (Search box)
    # We'll use coordinates from the first screenshot or just guess a central area for a basic test
    # In a real scenario, we'd parse the 'elements' list.
    print("\n--- Step 2: Coordinate Click ---")
    elements = nav_res.get("elements", [])
    search_box = next((e for e in elements if "Search" in e["text"]), None)
    
    if search_box:
        test_step("click_search", {
            "action": "click",
            "x": search_box["x"],
            "y": search_box["y"],
            "session_id": session_id,
            "tab_id": "tab1"
        })
    else:
        print("Search box not found in elements, trying default center click")
        test_step("click_center", {
            "action": "click",
            "x": 960, "y": 540,
            "session_id": session_id,
            "tab_id": "tab1"
        })

    # 3. Type
    print("\n--- Step 3: Typing ---")
    test_step("type_text", {
        "action": "type",
        "text": "Artificial Intelligence",
        "x": 960, "y": 540, # Assuming we clicked the center/search
        "session_id": session_id,
        "tab_id": "tab1"
    })
    
    test_step("press_enter", {
        "action": "enter",
        "x": 960, "y": 540,
        "session_id": session_id,
        "tab_id": "tab1"
    })

    # 4. Scroll
    print("\n--- Step 4: Scrolling ---")
    test_step("scroll_down", {
        "action": "scroll_down",
        "session_id": session_id,
        "tab_id": "tab1"
    })
    time.sleep(1)
    test_step("scroll_up", {
        "action": "scroll_up",
        "session_id": session_id,
        "tab_id": "tab1"
    })

    # 5. Tabs
    print("\n--- Step 5: Tab Management ---")
    test_step("navigate_tab2", {
        "url": "https://www.google.com",
        "session_id": session_id,
        "tab_id": "tab2"
    }, endpoint="/navigate")
    
    test_step("switch_back_tab1", {
        "session_id": session_id,
        "tab_id": "tab1"
    }, endpoint="/switch_tab")

    # 6. Drag and Drop (Simulated on a page that might support it, or just test the API)
    print("\n--- Step 6: Drag and Drop ---")
    test_step("drag_drop", {
        "action": "drag_and_drop",
        "x": 100, "y": 100,
        "end_x": 500, "end_y": 500,
        "session_id": session_id,
        "tab_id": "tab1"
    })

    print("\nValidation complete. Check aeon_output/browser_validation for results.")

if __name__ == "__main__":
    main()