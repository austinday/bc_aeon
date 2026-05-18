import requests
import json
import sys

def diagnose():
    url = "http://localhost:8188"
    print(f"Diagnosing ComfyUI at {url}...")
    
    try:
        # 1. Check basic connectivity
        resp = requests.get(f"{url}/system", timeout=5)
        print(f"System check: {resp.status_code}")
        
        # 2. Get all available nodes
        # The /object/status endpoint usually provides the list of available node types
        print("Fetching available nodes...")
        node_resp = requests.get(f"{url}/object/status", timeout=10)
        if node_resp.status_code == 200:
            nodes = node_resp.json()
            # The response is usually a list of lists or a dict
            # We want to see the class names
            all_classes = []
            if isinstance(nodes, list):
                for item in nodes:
                    if isinstance(item, list) and len(item) > 0:
                        all_classes.append(str(item[0]))
            elif isinstance(nodes, dict):
                all_classes = list(nodes.keys())
            
            print(f"Found {len(all_classes)} node classes.")
            
            # Search for LTX related nodes
            ltx_nodes = [n for n in all_classes if "LTX" in n.upper()]
            print("\n--- LTX Related Nodes ---")
            for n in sorted(ltx_nodes):
                print(n)
            
            with open("scripts/debug/live_nodes_current.json", "w") as f:
                json.dump(all_classes, f, indent=2)
            print("\nFull node list saved to scripts/debug/live_nodes_current.json")
            
        else:
            print(f"Failed to fetch nodes: {node_resp.status_code}")
            print(f"Response: {node_resp.text}")

    except Exception as e:
        print(f"Error during diagnosis: {e}")

if __name__ == "__main__":
    diagnose()