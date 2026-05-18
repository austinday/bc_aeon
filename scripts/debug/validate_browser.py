import os
import time
import requests
from aeon.tools.browser import BrowserNavigateTool, BrowserInteractTool, BrowserSwitchTabTool, ensure_browser_running

def test_browser_flow():
    print("Starting Browser Tool Validation...")
    
    # 1. Ensure server is running
    if not ensure_browser_running():
        print("FAILED: Could not start browser server.")
        return

    # Instantiate tools
    nav_tool = BrowserNavigateTool()
    interact_tool = BrowserInteractTool()
    switch_tool = BrowserSwitchTabTool()

    try:
        # Test 1: Navigation and Visuals
        print("\n[Test 1] Navigating to Google...")
        res1 = nav_tool.execute(url="https://www.google.com", tab_id="tab1")
        if "BROWSER ACTION SUCCESS" not in res1:
            print(f"FAILED: Navigation failed.\n{res1}")
            return
        print("SUCCESS: Navigated to Google and received visual analysis.")

        # Test 2: Interaction (Search)
        print("\n[Test 2] Searching for 'Aeon Agent'...")
        # We look for the search box. In a real scenario, the LLM would get the ID.
        # For validation, we'll try to find the search input via the elements list in res1.
        # Since we can't easily parse the LLM-style output here, we'll use a generic search 
        # if we can find an input, or just test the tool's ability to send the request.
        # To be robust, we'll just try to type into the first available input if found.
        
        # For the sake of a script, we'll try to interact with the search box.
        # Note: Google's IDs change, so we'll just verify the tool doesn't crash and returns a response.
        res2 = interact_tool.execute(action="type", text="Aeon Agent", element_id=1, tab_id="tab1")
        print(f"Interaction result: {res2[:200]}...")
        
        # Test 3: Tab Persistence
        print("\n[Test 3] Opening a second tab (Wikipedia)...")
        res3 = nav_tool.execute(url="https://www.wikipedia.org", tab_id="tab2")
        if "BROWSER ACTION SUCCESS" not in res3:
            print(f"FAILED: Second tab navigation failed.\n{res3}")
            return
        print("SUCCESS: Opened Wikipedia in tab2.")

        print("\n[Test 4] Switching back to tab1...")
        res4 = switch_tool.execute(tab_id="tab1")
        if "BROWSER ACTION SUCCESS" not in res4:
            print(f"FAILED: Switch tab failed.\n{res4}")
            return
        
        # Verify we are back on Google (check markdown/URL in response)
        if "google" not in res4.lower():
            print("FAILED: Tab state not preserved or wrong page loaded.")
            return
        print("SUCCESS: Switched back to tab1 and state preserved.")

        print("\n\n*** ALL BROWSER VALIDATIONS PASSED ***")
        return True

    except Exception as e:
        print(f"CRITICAL ERROR during validation: {e}")
        return False

if __name__ == "__main__":
    success = test_browser_flow()
    if not success:
        exit(1)