import requests
import json

URL = "http://localhost:8188"

def test():
    print(f"Testing connectivity to {URL}...")
    try:
        # Test root
        r = requests.get(f"{URL}/", timeout=5)
        print(f"Root / response: {r.status_code}")
        
        # Test API nodes (the correct endpoint for ComfyUI is /object_info)
        r = requests.get(f"{URL}/object_info", timeout=5)
        print(f"Object info /object_info response: {r.status_code}")
        if r.status_code == 200:
            nodes = r.json()
            ltx_nodes = [k for k in nodes.keys() if "LTX" in k]
            print(f"Found {len(ltx_nodes)} LTX-related nodes.")
            print("LTX Nodes:", ltx_nodes)
            
            # Save to file for analysis
            with open("scripts/debug/live_nodes_verified.json", "w") as f:
                json.dump(nodes, f, indent=2)
            print("Nodes saved to scripts/debug/live_nodes_verified.json")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()