import os
import sys
import requests

# Add project root to sys.path to allow importing aeon
PROJECT_ROOT = "/home/aday/bc_aeon"
sys.path.append(PROJECT_ROOT)

try:
    from aeon.tools.browser import BrowserNavigateTool, ensure_browser_running
    print("[Verify] Successfully imported browser tools.")
except ImportError as e:
    print(f"[Verify] Import failed: {e}")
    sys.exit(1)

def test_portability():
    print(f"[Verify] Current Working Directory: {os.getcwd()}")
    
    try:
        print("[Verify] Attempting to ensure browser is running...")
        # This calls start_browser.sh using an absolute path derived from the file location
        ensure_browser_running()
        print("[Verify] ensure_browser_running() completed without error.")
        
        # Test actual API connectivity
        print("[Verify] Checking browser health endpoint...")
        res = requests.get("http://localhost:8030/health", timeout=5)
        if res.status_code == 200:
            print("[Verify] Browser service is healthy and responding!")
        else:
            print(f"[Verify] Browser service returned non-200: {res.status_code}")
            sys.exit(1)
            
        # Test a simple navigation
        print("[Verify] Testing navigation via BrowserNavigateTool...")
        tool = BrowserNavigateTool()
        result = tool.execute("https://www.google.com")
        
        if "BROWSER ACTION SUCCESS" in result:
            print("[Verify] Navigation successful!")
            print(result[:200] + "...")
        else:
            print(f"[Verify] Navigation failed: {result}")
            sys.exit(1)
            
        print("\n[RESULT] Portability test PASSED!")
        
    except Exception as e:
        print(f"[Verify] Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_portability()