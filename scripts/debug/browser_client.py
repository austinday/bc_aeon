import requests
import base64
import argparse
import sys
import json

URL = "http://localhost:8030"

def navigate(url, session_id):
    print(f"[*] Navigating to {url}...")
    res = requests.post(f"{URL}/navigate", json={
        "url": url,
        "session_id": session_id,
        "tab_id": "default"
    }, timeout=60)
    if res.status_code != 200:
        print(f"[-] HTTP Error: {res.status_code}")
        sys.exit(1)
    return res.json()

def interact(action, session_id, element_id=None, text=None):
    print(f"[*] Performing action {action} on element {element_id}...")
    payload = {
        "action": action,
        "session_id": session_id,
        "tab_id": "default"
    }
    if element_id is not None:
        payload["element_id"] = element_id
    if text is not None:
        payload["text"] = text
        
    res = requests.post(f"{URL}/interact", json=payload, timeout=60)
    if res.status_code != 200:
        print(f"[-] HTTP Error: {res.status_code}")
        sys.exit(1)
    return res.json()

def save_image(b64_str, filename):
    with open(filename, "wb") as f:
        f.write(base64.b64decode(b64_str))
    print(f"[+] Screenshot saved to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL to navigate to")
    parser.add_argument("--interact", help="Action to perform (click, type, scroll_down, etc.)")
    parser.add_argument("--element_id", type=int, help="ID of the element to interact with")
    parser.add_argument("--text", help="Text to type")
    parser.add_argument("--session", default="default_session", help="Session ID")
    parser.add_argument("--save", help="Filename to save screenshot")
    args = parser.parse_args()

    data = None
    if args.url:
        data = navigate(args.url, args.session)
    elif args.interact:
        data = interact(args.interact, args.session, args.element_id, args.text)
    else:
        print("[-] No action specified.")
        sys.exit(1)

    if data.get("status") == "success":
        if args.save:
            save_image(data["clean_b64"], args.save)
        
        print("\n--- Elements Found ---")
        for el in data.get("elements", []):
            print(f"ID: {el['id']} | Tag: {el['tag']} | Text: {el['text']}")
        print("----------------------\n")
    else:
        print(f"[-] API Error: {data}")
        sys.exit(1)

if __name__ == "__main__":
    main()