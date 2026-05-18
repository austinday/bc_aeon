import requests
import json

def query_node_info(node_name):
    url = "http://localhost:8188/object_info"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if node_name in data:
            node_info = data[node_name]
            print(f"--- Info for Node: {node_name} ---")
            print(json.dumps(node_info, indent=2))
            return node_info
        else:
            print(f"Node {node_name} not found in ComfyUI object_info.")
            return None
    except Exception as e:
        print(f"Error querying ComfyUI: {e}")
        return None

if __name__ == "__main__":
    # Target the specific node we are having trouble with
    query_node_info("EmptyLTXVLatentVideo")