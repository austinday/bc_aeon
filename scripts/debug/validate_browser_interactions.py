import requests
import time
import subprocess
import os
import sys

PORT = 8030
URL = f"http://localhost:{PORT}"

def start_browser():
    print("[*] Starting browser service...")
    # Use the same path as test_browser.py
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../aeon/scripts/start_browser.sh"))
    if not os.path.exists(script_path):
        # Fallback for different directory structures
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../aeon/scripts/start_browser.sh"))
    
    process = subprocess.Popen(["bash", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for health check
    for i in range(30):
        try:
            res = requests.get(f"{URL}/health", timeout=2)
            if res.status_code == 200:
                print("[+] Browser service is UP.")
                return process
        except:
            pass
        time.sleep(1)
    
    print("[-] Browser service failed to start.")
    process.kill()
    sys.exit(1)

def test_interactions():
    session_id = "val_session_123"
    
    # 1. Navigate
    print("\n[1] Testing Navigation...")
    nav_res = requests.post(f"{URL}/navigate", json={
        "url": "https://www.google.com", 
        "session_id": session_id
    }, timeout=30)
    
    if nav_res.status_code != 200:
        print(f"[-] Navigate failed: {nav_res.text}")
        return False
    
    data = nav_res.json()
    if data.get("status") != "success":
        print(f"[-] Navigate API error: {data}")
        return False
    
    elements = data.get("elements", [])
    if not elements:
        print("[-] No elements found on page.")
        return False
    
    print(f"[+] Successfully navigated. Found {len(elements)} elements.")
    target_el = elements[0]
    print(f"[+] Target element for tests: ID {target_el['id']} - {target_el['text']}")

    # 2. Test Element Persistence (Interact using element_id)
    print("\n[2] Testing Element Persistence (Clicking element_id)...")
    int_res = requests.post(f"{URL}/interact", json={
        "action": "click",
        "element_id": target_el['id'],
        "session_id": session_id
    }, timeout=30)
    
    if int_res.status_code != 200 or int_res.json().get("status") != "success":
        print(f"[-] Element persistence failed: {int_res.text}")
        return False
    print("[+] Element persistence verified!")

    # 3. Test Hover
    print("\n[3] Testing Hover...")
    hover_res = requests.post(f"{URL}/interact", json={
        "action": "hover",
        "element_id": target_el['id'],
        "session_id": session_id
    }, timeout=30)
    
    if hover_res.status_code != 200 or hover_res.json().get("status") != "success":
        print(f"[-] Hover failed: {hover_res.text}")
        return False
    print("[+] Hover action executed successfully!")

    # 4. Test Right Click
    print("\n[4] Testing Right Click...")
    rc_res = requests.post(f"{URL}/interact", json={
        "action": "right_click",
        "element_id": target_el['id'],
        "session_id": session_id
    }, timeout=30)
    
    if rc_res.status_code != 200 or rc_res.json().get("status") != "success":
        print(f"[-] Right click failed: {rc_res.text}")
        return False
    print("[+] Right click action executed successfully!")

    return True

if __name__ == "__main__":
    browser_proc = start_browser()
    try:
        success = test_interactions()
        if success:
            print("\n" + "="*30 + "\nALL BROWSER TESTS PASSED\n" + "="*30)
            sys.exit(0)
        else:
            print("\n" + "!"*30 + "\nSOME BROWSER TESTS FAILED\n" + "!"*30)
            sys.exit(1)
    finally:
        browser_proc.terminate()