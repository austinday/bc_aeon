import json
import os

info_path = 'aeon_output/debug/comfyui_object_info.json'

if not os.path.exists(info_path):
    print(f"Error: {info_path} not found.")
    exit(1)

with open(info_path, 'r') as f:
    data = json.load(f)

print(f"{'Node Name':<40} | {'Inputs':<50} | {'Outputs'}")
print("-" * 110)

for node_name, node_info in data.items():
    # Check if it outputs CONDITIONING
    outputs = node_info.get('output', [])
    if 'CONDITIONING' not in outputs:
        continue
    
    # Check if it takes CLIP as input
    inputs = node_info.get('input', {}).get('required', {})
    has_clip = False
    input_str = ""
    for inp_name, inp_type in inputs.items():
        # Handle list types in the JSON
        actual_type = inp_type[0] if isinstance(inp_type, list) else inp_type
        if actual_type == 'CLIP':
            has_clip = True
        input_str += f"{inp_name}:{actual_type}, "
    
    if has_clip and node_name != 'CLIPTextEncode':
        print(f"{node_name:<40} | {input_str.strip(', '):<50} | {', '.join(outputs)}")
