import json

with open('scripts/debug/ltx_nodes_details.json', 'r') as f:
    nodes = json.load(f)

print("Searching for nodes related to 'projection' or 'encoder'...")
for node_name, details in nodes.items():
    # Check node name
    if 'projection' in node_name.lower() or 'encoder' in node_name.lower():
        print(f"\n--- Node: {node_name} ---")
        print(f"Inputs: {json.dumps(details.get('input', {}), indent=2)}")
        print(f"Outputs: {details.get('output', [])}")
    else:
        # Check if any input contains 'projection' or 'encoder'
        inputs = details.get('input', {})
        if 'required' in inputs:
            for req_name in inputs['required']:
                if 'projection' in req_name.lower() or 'encoder' in req_name.lower():
                    print(f"\n--- Node: {node_name} (found in inputs) ---")
                    print(f"Inputs: {json.dumps(inputs, indent=2)}")
                    print(f"Outputs: {details.get('output', [])}")
                    break