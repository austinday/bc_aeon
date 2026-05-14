import json
import os

def find_connector():
    nodes_info_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(nodes_info_path):
        print(f"Error: {nodes_info_path} not found.")
        return

    print(f"Loading nodes info from {nodes_info_path}...")
    with open(nodes_info_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return

    search_terms = ["connector", "projection", "ltxv", "embedding"]
    found_nodes = []

    for node_class, info in data.items():
        if info is None:
            continue
        
        # Safely get display name and description
        display_name = str(info.get("display_name") or "").lower()
        description = str(info.get("description") or "").lower()
        
        if any(term in display_name or term in description or term in node_class.lower() for term in search_terms):
            found_nodes.append({
                "class": node_class,
                "display_name": info.get("display_name"),
                "description": info.get("description"),
                "inputs": info.get("inputs")
            })

    if not found_nodes:
        print("No matching nodes found.")
    else:
        print(f"Found {len(found_nodes)} potential nodes:\n")
        for node in found_nodes:
            print(f"Class: {node['class']}")
            print(f"Display Name: {node['display_name']}")
            print(f"Description: {node['description']}")
            print(f"Inputs: {node['inputs']}")
            print("-" * 40)

if __name__ == "__main__":
    find_connector()