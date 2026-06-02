import requests
import json
import time

# The server is running on port 8080
BASE_URL = "http://localhost:8080"

def test_press_and_hold():
    session_id = "test_session"
    
    # 1. Navigate to a data URI that tests press-and-hold
    # We use a raw string for the HTML to avoid f-string brace issues
    html_content = """
    <html>
    <body style='margin:0; padding:0;'>
        <div id='target' style='width:100px; height:100px; background:red; color:white; display:flex; align-items:center; justify-content:center; cursor:pointer; font-family:sans-serif;'>
            Hold Me
        </div>
        <script>
            const el = document.getElementById('target');
            let start = 0;
            el.onpointerdown = () => { 
                console.log('Pointer Down');
                start = Date.now(); 
            };
            el.onpointerup = () => { 
                console.log('Pointer Up');
                let duration = Date.now() - start;
                if(duration > 1000) { 
                    el.innerText = 'Held!'; 
                } 
            };
        </script>
    </body>
    </html>
    """
    url = f"data:text/html;base64,{__import__('base64').b64encode(html_content.encode()).decode()}"
    
    print(f"Navigating to test page...")
    try:
        resp = requests.post(f"{BASE_URL}/navigate", json={"session_id": session_id, "url": url})
        resp.raise_for_status()
    except Exception as e:
        print(f"Navigation failed: {e}")
        return

    # 2. Use the debug endpoint to perform press_and_hold on the #target selector
    print("Performing press_and_hold for 2000ms...")
    try:
        payload = {
            "session_id": session_id,
            "action": "press_and_hold",
            "selector": "#target",
            "duration": 2000
        }
        resp = requests.post(f"{BASE_URL}/debug_press_and_hold", json=payload)
        resp.raise_for_status()
    except Exception as e:
        print(f"Interaction failed: {e}")
        return

    # 3. Verify the text changed to 'Held!'
    print("Verifying result...")
    try:
        payload = {
            "session_id": session_id,
            "action": "get_text",
            "selector": "#target"
        }
        resp = requests.post(f"{BASE_URL}/debug_get_text", json=payload)
        resp.raise_for_status()
        text = resp.json().get("text", "")
        print(f"Element text: '{text}'")
        if text == "Held!":
            print("SUCCESS: press_and_hold triggered the 'Held!' state.")
        else:
            print("FAILURE: Element text did not change to 'Held!'.")
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    test_press_and_hold()