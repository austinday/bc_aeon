import json
import os

json_path = "aeon_output/debug/comfyui_object_info.json"

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    exit(1)

with open(json_path, "r") as f:
    data = json.load(f)

print(f"{'Node Class':<40} | {'Inputs':<30} | {'Outputs':<30}")
print("-" * 105)

found_any = False
for node_class, info in data.items():
    # Check outputs
    outputs = info.get("output", [])
    if not isinstance(outputs, list):
        outputs = [outputs]
    
    if "LATENT" in outputs:
        # Check inputs
        inputs = info.get("input", {}).get("required", {})
        input_types = []
        for input_name, input_info in inputs.items():
            if isinstance(input_info, list) and len(input_info) > 0:
                input_types.append(input_info[0])
        
        if "IMAGE" in input_types:
            found_any = True
            print(f"{node_class:<40} | {', '.join(input_types):<30} | {', '.join(outputs):<30}")

if not found_any:
    print("No nodes found that take IMAGE and output LATENT.")