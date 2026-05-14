import json
import os

def find_ltx_t5_conditioning():
    info_path = 'aeon_output/debug/comfyui_object_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        nodes = json.load(f)

    results = []
    for node_name, node_info in nodes.items():
        # Check if node outputs CONDITIONING
        outputs = node_info.get('output', [])
        if 'CONDITIONING' in outputs:
            # Check if node name contains LTX or T5 (case insensitive)
            if 'LTX' in node_name.upper() or 'T5' in node_name.upper():
                results.append({
                    'name': node_name,
                    'inputs': node_info.get('input', {}),
                    'outputs': outputs
                })

    if not results:
        print("No nodes found that output CONDITIONING and contain 'LTX' or 'T5' in their name.")
        return

    print(f"{'Node Name':<40} | {'Inputs'}")
    print("-" * 100)
    for res in results:
        inputs = res['inputs']
        input_str = ""
        if 'required' in inputs:
            req = inputs['required']
            input_str = ", ".join([f"{k}: {v[0]}" for k, v in req.items()])
        
        print(f"{res['name']:<40} | {input_str}")

if __name__ == "__main__":
    find_ltx_t5_conditioning()