import json
import os

def analyze_nodes(json_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # We are looking for nodes that are likely related to LTX or Video generation/manipulation
    keywords = ['LTX', 'Video', 'Latent', 'Sampler', 'VAE', 'VHS']
    relevant_nodes = {}

    for node_type, info in data.items():
        if any(kw.lower() in node_type.lower() for kw in keywords):
            relevant_nodes[node_type] = {
                "inputs": info.get("inputs", {}),
                "outputs": info.get("outputs", {})
            }

    print(f"Found {len(relevant_nodes)} potentially relevant nodes.\n")
    
    # Specifically look for LTX-Video nodes
    print("=== LTX-Specific Nodes ===")
    ltx_nodes = [n for n in relevant_nodes if 'LTX' in n]
    for node in sorted(ltx_nodes):
        print(f"\nNode: {node}")
        print(f"  Inputs: {list(relevant_nodes[node]['inputs'].keys())}")
        print(f"  Outputs: {list(relevant_nodes[node]['outputs'].keys())}")

    # Look for Video Helper Suite (VHS) nodes which are crucial for loading/saving/combining
    print("\n=== VHS (Video Helper Suite) Nodes ===")
    vhs_nodes = [n for n in relevant_nodes if 'VHS' in n]
    for node in sorted(vhs_nodes):
        print(f"\nNode: {node}")
        print(f"  Inputs: {list(relevant_nodes[node]['inputs'].keys())}")
        print(f"  Outputs: {list(relevant_nodes[node]['outputs'].keys())}")

    # Look for general Latent manipulation nodes
    print("\n=== Latent Manipulation Nodes ===")
    latent_nodes = [n for n in relevant_nodes if 'Latent' in n and 'LTX' not in n]
    for node in sorted(latent_nodes):
        print(f"\nNode: {node}")
        print(f"  Inputs: {list(relevant_nodes[node]['inputs'].keys())}")
        print(f"  Outputs: {list(relevant_nodes[node]['outputs'].keys())}")

if __name__ == "__main__":
    analyze_nodes("aeon_output/debug/comfyui_nodes_info.json")