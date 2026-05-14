import json
import os

INFO_PATH = 'aeon_output/debug/comfyui_object_info.json'

if not os.path.exists(INFO_PATH):
    print(f"Error: {INFO_PATH} not found.")
    exit(1)

with open(INFO_PATH, 'r') as f:
    data = json.load(f)

print(f"Analyzing {len(data)} nodes...\n")

candidates = []

for node_name, info in data.items():
    # We are looking for nodes that output 'CONDITIONING'
    outputs = info.get('output', [])
    if not outputs or 'CONDITIONING' not in str(outputs).upper():
        continue
    
    # We are specifically interested in LTX related nodes
    if 'LTX' in node_name.upper() or 'T5' in node_name.upper():
        inputs = info.get('input', {})
        required_inputs = inputs.get('required', {})
        
        # Check if it takes STRING or CLIP as input
        input_types = []
        for inp_name, inp_info in required_inputs.items():
            # inp_info is usually a list [type, {details}]
            if isinstance(inp_info, list) and len(inp_info) > 0:
                input_types.append(str(inp_info[0]))
            else:
                input_types.append(str(inp_info))

        candidates.append({
            "node": node_name,
            "inputs": input_types,
            "outputs": outputs,
            "info": info
        })

if not candidates:
    print("No LTX/T5 nodes found that output CONDITIONING.")
else:
    print(f"Found {len(candidates)} candidates:\n")
    for c in candidates:
        print(f"Node: {c['node']}")
        print(f"  Inputs: {c['inputs']}")
        print(f"  Outputs: {c['outputs']}")
        print("-" * 30)

# Also search for any node that takes STRING and outputs CONDITIONING, regardless of name
print("\nSearching for any STRING -> CONDITIONING nodes (generic):")
generic_candidates = []
for node_name, info in data.items():
    outputs = info.get('output', [])
    if 'CONDITIONING' in str(outputs).upper():
        inputs = info.get('input', {}).get('required', {})
        for inp_name, inp_info in inputs.items():
            if isinstance(inp_info, list) and 'STRING' in str(inp_info[0]).upper():
                generic_candidates.append(node_name)
                break

print(f"Generic STRING -> CONDITIONING nodes: {generic_candidates}")