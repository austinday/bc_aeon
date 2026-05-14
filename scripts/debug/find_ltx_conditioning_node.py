import json
import os

# Path to the object info file
INFO_FILE = "aeon_output/debug/comfyui_object_info.json"

if not os.path.exists(INFO_FILE):
    print(f"Error: {INFO_FILE} not found.")
    exit(1)

with open(INFO_FILE, "r") as f:
    data = json.load(f)

print(f"Searching for LTX/T5 nodes that output 'CONDITIONING'...")
print("-" * 60)

found_count = 0
for node_name, info in data.items():
    # Check if node name contains LTX or T5 (case insensitive)
    if "LTX" in node_name.upper() or "T5" in node_name.upper():
        outputs = info.get("output", [])
        if "CONDITIONING" in outputs:
            inputs = info.get("input", {}).get("required", {})
            # We are looking for nodes that take CLIP or STRING as input
            input_types = []
            for in_name, in_info in inputs.items():
                # in_info is usually a list [type, {details}]
                if isinstance(in_info, list) and len(in_info) > 0:
                    input_types.append(in_info[0])
                elif isinstance(in_info, str):
                    input_types.append(in_info)
            
            if "CLIP" in input_types or "STRING" in input_types:
                found_count += 1
                print(f"NODE: {node_name}")
                print(f"  Inputs: {inputs}")
                print(f"  Outputs: {outputs}")
                print("-" * 60)

if found_count == 0:
    print("No matching nodes found. Expanding search to all nodes outputting CONDITIONING...")
    for node_name, info in data.items():
        outputs = info.get("output", [])
        if "CONDITIONING" in outputs:
            inputs = info.get("input", {}).get("required", {})
            input_types = []
            for in_name, in_info in inputs.items():
                if isinstance(in_info, list) and len(in_info) > 0:
                    input_types.append(in_info[0])
                elif isinstance(in_info, str):
                    input_types.append(in_info)
            
            if "CLIP" in input_types or "STRING" in input_types:
                print(f"NODE: {node_name} (Generic) | Inputs: {input_types} | Outputs: {outputs}")
