import json
import os

def find_nodes():
    info_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(info_path):
        print(f"File not found: {info_path}")
        return

    print(f"Loading {info_path}...")
    with open(info_path, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return

    # The structure of comfyui_nodes_info.json can vary. 
    # It's usually a dict of node classes or a list.
    
    found_nodes = []
    
    # Search in keys (class names) and values (descriptions/inputs)
    if isinstance(data, dict):
        for node_class, details in data.items():
            if "LTXV" in node_class.upper() or "CONNECTOR" in node_class.upper():
                found_nodes.append((node_class, details))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("class_type", "")
                if "LTXV" in name.upper() or "CONNECTOR" in name.upper():
                    found_nodes.append((name, item))

    if not found_nodes:
        print("No LTXV or Connector nodes found.")
        return

    print(f"\nFound {len(found_nodes)} matching nodes:\n")
    for name, details in found_nodes:
        print(f"--- Node: {name} ---")
        print(json.dumps(details, indent=2))
        print("\n")

if __name__ == "__main__":
    find_nodes()