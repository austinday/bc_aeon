import requests
import json
import sys

def main():
    url = "http://localhost:8188/object_info"
    print(f"Fetching node information from {url}...")
    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        data = res.json()
        
        output_path = "aeon_output/debug/comfyui_nodes_info.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Successfully dumped node info to {output_path}")
        print(f"Total nodes found: {len(data)}")
        
    except Exception as e:
        print(f"Error fetching node info: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()