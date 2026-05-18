import json
import os

def inspect_ltx_nodes():
    info_path = 'aeon_output/debug/comfyui_nodes_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    ltx_nodes = {}
    for node_name, node_info in data.items():
        # Look for LTX in the name or any of the input/output types
        is_ltx = 'LTX' in node_name.upper()
        
        # Also check if it's a known LTX-related node from previous searches
        if is_ltx:
            ltx_nodes[node_name] = node_info

    print(f"Found {len(ltx_nodes)} LTX-related nodes.\n")
    
    for name, info in ltx_nodes.items():
        print(f"--- Node: {name} ---")
        # Print inputs
        inputs = info.get('input', {})
        required = inputs.get('required', {})
        print("  Inputs (Required):")
        for inp, details in required.items():
            print(f"    - {inp}: {details}")
        
        # Print outputs
        outputs = info.get('output', [])
        print(f"  Outputs: {outputs}")
        print("\n")

if __name__ == "__main__":
    inspect_ltx_nodes()