import json
import os

def analyze_nodes():
    info_path = "aeon_output/debug/comfyui_object_info.json"
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            return

    results = []
    for node_name, info in data.items():
        # Check if it outputs CONDITIONING
        outputs = info.get('output', [])
        if 'CONDITIONING' in outputs:
            # Check for LTX or T5 in name or description
            name_lower = node_name.lower()
            desc_lower = info.get('description', '').lower()
            
            # Also check if it takes STRING as input
            inputs = info.get('input', {}).get('required', {})
            takes_string = any('STRING' in str(val) for val in inputs.values())
            
            if 'ltx' in name_lower or 't5' in name_lower or 'ltx' in desc_lower or 't5' in desc_lower or takes_string:
                results.append({
                    'name': node_name,
                    'description': info.get('description', 'No description'),
                    'inputs': inputs,
                    'outputs': outputs
                })

    print(f"Found {len(results)} potential conditioning nodes:\n")
    for res in results:
        print(f"Node: {res['name']}")
        print(f"  Desc: {res['description']}")
        print(f"  Inputs: {res['inputs']}")
        print(f"  Outputs: {res['outputs']}")
        print("-" * 40)

if __name__ == "__main__":
    analyze_nodes()