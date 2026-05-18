import json
import os

def find_connector():
    json_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return

    keywords = ["connector", "projection", "project", "embedding_connector"]
    found_nodes = []

    for node_class, info in data.items():
        # Convert everything to string to search across all fields
        node_str = str(info).lower()
        if any(kw in node_str for kw in keywords):
            found_nodes.append((node_class, info))

    if not found_nodes:
        print("No nodes found matching the keywords.")
        return

    print(f"Found {len(found_nodes)} potential nodes:\n")
    for node_class, info in found_nodes:
        print(f"Class: {node_class}")
        print(f"Info: {json.dumps(info, indent=2)}")
        print("-" * 40)

if __name__ == "__main__":
    find_connector()