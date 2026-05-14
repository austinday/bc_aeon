import json
import os

json_path = "aeon_output/debug/comfyui_nodes_info.json"

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    exit(1)

with open(json_path, "r") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        exit(1)

print(f"Analyzing {len(data)} nodes...")

matches = []
for node_name, info in data.items():
    # Check outputs
    outputs = info.get("outputs", [])
    has_conditioning_output = any(out.get("type") == "CONDITIONING" for out in outputs)
    
    # Check inputs
    inputs = info.get("input", {}) # Some nodes use 'input', some use 'inputs'
    if not inputs:
        inputs = info.get("inputs", {})
        
    # The inputs structure can be a dict of {name: [type]} or {name: type}
    has_string_input = False
    if isinstance(inputs, dict):
        for input_name, input_val in inputs.items():
            if isinstance(input_val, list):
                if any(t == "STRING" for t in input_val):
                    has_string_input = True
                    break
            elif input_val == "STRING":
                has_string_input = True
                break
    
    if has_conditioning_output and has_string_input:
        matches.append(node_name)

if matches:
    print(f"Found {len(matches)} nodes that take STRING and output CONDITIONING:")
    for m in matches:
        print(f"- {m}")
else:
    print("No nodes found matching the criteria.")

# Also search for any node that mentions 'LTX' and outputs 'CONDITIONING'
print("\nSearching for any LTX nodes that output CONDITIONING...")
ltx_cond = []
for node_name, info in data.items():
    outputs = info.get("outputs", [])
    if any(out.get("type") == "CONDITIONING" for out in outputs):
        if "LTX" in node_name.upper():
            ltx_cond.append(node_name)

if ltx_cond:
    print(f"Found {len(ltx_cond)} LTX nodes with CONDITIONING output:")
    for m in ltx_cond:
        print(f"- {m}")
else:
    print("No LTX nodes found with CONDITIONING output.")