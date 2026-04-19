import os
import sys
import time
import requests
import subprocess
import base64

PORT = 8030
URL = f"http://localhost:{PORT}"

def run_tests():
    print("=" * 60)
    print("Starting Advanced Stealth Browser Test")
    print("=" * 60)
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aeon", "scripts", "start_browser.sh"))
    
    print("\n[1] Starting browser service via start_browser.sh...")
    res = subprocess.run(["bash", script_path], capture_output=True, text=True)
    print(res.stdout.strip())
    
    print("\n[2] Checking /health endpoint manually...")
    try:
        health = requests.get(f"{URL}/health", timeout=5)
        print(f"Health response: {health.status_code} - {health.text}")
    except Exception as e:
        print(f"\n❌ FAILED: Could not reach health endpoint. Error: {e}")
        sys.exit(1)

    print("\n[3] Testing /navigate to https://www.bananacoconut.com...")
    session_id = "stealth_test_session_1"
    try:
        nav_res = requests.post(
            f"{URL}/navigate",
            json={"url": "https://www.bananacoconut.com", "session_id": session_id},
            timeout=60
        )
        if nav_res.status_code == 200:
            data = nav_res.json()
            if data.get("status") == "success":
                print("✅ Navigate successful!")
                print(f"   Extracted {len(data.get('elements', []))} interactive elements.")
                
                # Print the markdown to prove we bypassed Cloudflare (if CF blocked us, it would say "Just a moment...")
                markdown = data.get('markdown', '').strip()
                print(f"   Markdown preview:\n   ----------------------------------------\n   {markdown[:500]}...\n   ----------------------------------------")
                
                # Save the image locally so you can open it and verify with your own eyes!
                with open("aeon_output/test_bananacoconut_top.jpg", "wb") as f:
                    f.write(base64.b64decode(data["clean_b64"]))
                print("   📸 Saved screenshot to 'aeon_output/test_bananacoconut_top.jpg'")
            else:
                print(f"❌ Navigate returned API error status: {data}")
        else:
            print(f"❌ Navigate HTTP failed with status {nav_res.status_code}: {nav_res.text}")
    except Exception as e:
        print(f"❌ Navigate request failed: {e}")
        
    print("\n[4] Testing /interact (Scrolling down)...")
    try:
        int_res = requests.post(
            f"{URL}/interact",
            json={"action": "scroll_down", "session_id": session_id},
            timeout=60
        )
        if int_res.status_code == 200:
            data = int_res.json()
            if data.get("status") == "success":
                print("✅ Scroll successful!")
                with open("aeon_output/test_bananacoconut_scrolled.jpg", "wb") as f:
                    f.write(base64.b64decode(data["clean_b64"]))
                print("   📸 Saved scrolled screenshot to 'aeon_output/test_bananacoconut_scrolled.jpg'")
            else:
                print(f"❌ Interact returned API error status: {data}")
        else:
            print(f"❌ Interact HTTP failed with status {int_res.status_code}: {int_res.text}")
    except Exception as e:
        print(f"❌ Interact request failed: {e}")
    
    print("\n[5] Cleaning up session...")
    try:
        requests.post(f"{URL}/close_session", json={"session_id": session_id}, timeout=5)
        print("Session closed.")
    except Exception as e:
        print(f"Failed to close session: {e}")

if __name__ == "__main__":
    run_tests()
