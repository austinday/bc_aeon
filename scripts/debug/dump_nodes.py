import requests
import json
import sys

def main():
    url = "http://localhost:8188/object_info"
    print(f"Fetching node info from {url}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            with open("scripts/debug/live_nodes.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"Successfully dumped {len(data)} nodes to scripts/debug/live_nodes.json")
        else:
            print(f"Error: Server returned status {response.status_code}")
            print(f"Response text: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()