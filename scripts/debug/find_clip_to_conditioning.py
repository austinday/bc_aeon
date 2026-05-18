import json
import os

def find_clip_to_conditioning():
    info_path = 'aeon_output/debug/comfyui_object_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    results = []
    for node_name, node_info in data.items():
        # Check if it outputs CONDITIONING
        outputs = node_info.get('output', [])
        if 'CONDITIONING' not in outputs:
            continue
        
        # Check if it takes CLIP as input
        inputs = node_info.get('input', {}).get('required', {})
        has_clip_input = False
        for input_name, input_type in inputs.items():
            if isinstance(input_type, list) and 'CLIP' in input_type:
                has_clip_input = True
                break
        
        if has_clip_input:
            # Exclude standard CLIPTextEncode to find the LTX specific one
            if "CLIPTextEncode" in node_name:
                continue
            results.append({
                "node": node_name,
                "inputs": inputs,
                "outputs": outputs
            })

    if not results:
        print("No non-standard CLIP -> CONDITIONING nodes found.")
    else:
        print(f"Found {len(results)} potential CLIP -> CONDITIONING nodes:\n")
        for res in results:
            print(f"Node: {res['node']}")
            print(f"  Inputs: {res['inputs']}")
            print(f"  Outputs: {res['outputs']}")
            print("-" * 40)

if __name__ == "__main__":
    find_clip_to_conditioning()