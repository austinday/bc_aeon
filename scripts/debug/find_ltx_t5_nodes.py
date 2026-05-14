import json
import os

def find_nodes():
    info_path = 'aeon_output/debug/comfyui_object_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    print(f"Searching for LTX or T5 related nodes in {len(data)} nodes...\n")
    
    found_count = 0
    for node_name, info in data.items():
        # Search for LTX or T5 in the node name
        if 'LTX' in node_name.upper() or 'T5' in node_name.upper():
            found_count += 1
            print(f"=== NODE: {node_name} ===")
            
            # Print Inputs
            inputs = info.get('input', {})
            required = inputs.get('required', {})
            print(f"Inputs (Required): {list(required.keys())}")
            for k, v in required.items():
                print(f"  - {k}: {v}")
            
            # Print Outputs
            outputs = info.get('output', [])
            print(f"Outputs: {outputs}")
            print("-" * 40)

    if found_count == 0:
        print("No LTX or T5 related nodes found.")
    else:
        print(f"\nFound {found_count} candidate nodes.")

if __name__ == "__main__":
    find_nodes()