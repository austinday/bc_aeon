import requests
import os
import subprocess
import time

PORT = 8030
URL = f"http://localhost:{PORT}"
# We use a data URL to ensure the browser can load the content regardless of mount points
with open("scripts/debug/browser_stress_test.html", "r") as f:
    html_content = f.read()
import base64
data_url = f"data:text/html;base64,{base64.b64encode(html_content.encode()).decode()}"

def run_stress_test():
    print("Starting Browser Stress Test...")
    
    # 1. Start the browser service
    # Assuming start_browser.sh is available as seen in test_browser.py
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../aeon/scripts/start_browser.sh"))
    subprocess.run(["bash", script_path], capture_output=True)
    
    # Wait for startup
    time.sleep(5)
    
    session_id = "stress_test_session"
    
    # 2. Navigate to the stress test page
    print(f"\nNavigating to stress test page...")
    res = requests.post(f"{URL}/navigate", json={"url": data_url, "session_id": session_id})
    
    if res.status_code != 200:
        print(f"Failed to navigate: {res.text}")
        return

    data = res.json()
    elements = data.get("elements", [])
    
    print(f"\nFound {len(elements)} elements.")
    
    # Check for specific elements
    found_std = any("Standard Button" in e['text'] for e in elements)
    found_iframe = any("Iframe Button" in e['text'] for e in elements)
    found_shadow = any("Shadow Button" in e['text'] for e in elements)
    found_file = any(e['tag'] == 'input' for e in elements) # Basic check for input

    print(f"\n--- RESULTS ---")
    print(f"Standard Button detected: {'✅' if found_std else '❌'}")
    print(f"iFrame Button detected:  {'✅' if found_iframe else '❌'} (Expected: ✅)")
    print(f"Shadow Button detected: {'✅' if found_shadow else '❌'} (Expected: ✅)")
    print(f"File Input detected:      {'✅' if found_file else '❌'}")
    
    if not found_iframe or not found_shadow:
        print("\nCONCLUSION: The browser tool cannot see elements inside iFrames or Shadow DOM.")
    else:
        print("\nCONCLUSION: The browser tool surprisingly saw the hidden elements.")

    # Cleanup
    requests.post(f"{URL}/close_session", json={"session_id": session_id})

if __name__ == "__main__":
    run_stress_test()