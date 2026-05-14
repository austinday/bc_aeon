import json
import os

def find_connector_nodes():
    info_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        try:
            nodes = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return

    print(f"Scanning {len(nodes)} nodes for LTXV connector/embedding logic...\n")
    
    found_nodes = []
    keywords = ["connector", "embedding", "projection", "ltxv", "ltx-video"]
    
    for node_class, info in nodes.items():
        # Check class name, description, and inputs
        text_to_scan = (node_class + " " + (info.get("description", "") or "")).lower()
        inputs = str(info.get("inputs", [])).lower()
        
        if any(kw in text_to_scan for kw in keywords) or any(kw in inputs for kw in keywords):
            # We are specifically looking for something that likely takes a model/clip and returns conditioning
            # or something that loads a .safetensors connector file.
            found_nodes.append((node_class, info))

    if not found_nodes:
        print("No matching nodes found.")
        return

    print(f"Found {len(found_nodes)} potential nodes. Filtering for most relevant...\n")
    
    for node_class, info in found_nodes:
        # Prioritize nodes that look like they handle the embedding connector
        # LTXV usually needs a node to load the connector and then apply it.
        print(f"--- Node: {node_class} ---")
        print(f"Description: {info.get('description', 'N/A')}")
        print(f"Inputs: {info.get('inputs', 'N/A')}")
        print(f"Outputs: {info.get('outputs', 'N/A')}")
        print("-" * 40)

if __name__ == "__main__":
    find_connector_nodes()