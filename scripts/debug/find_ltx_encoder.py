import json
import os

def find_encoder():
    info_path = 'aeon_output/debug/comfyui_object_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    candidates = []
    for node_name, node_info in data.items():
        # Check if it outputs CONDITIONING
        outputs = node_info.get('output', [])
        if 'CONDITIONING' not in outputs:
            continue
        
        # Check if it takes STRING as input
        inputs = node_info.get('input', {}).get('required', {})
        has_string = False
        for input_name, input_type in inputs.items():
            # input_type is usually a list [type, {details}]
            if isinstance(input_type, list) and 'STRING' in input_type[0]:
                has_string = True
                break
        
        if has_string:
            candidates.append(node_name)

    print("Nodes that take STRING and output CONDITIONING:")
    for c in candidates:
        print(f"- {c}")

    # Specifically look for LTX or T5 in the name
    ltx_t5_candidates = [c for c in candidates if 'LTX' in c.upper() or 'T5' in c.upper()]
    print("\nLTX/T5 specific candidates:")
    for c in ltx_t5_candidates:
        print(f"- {c}")

if __name__ == "__main__":
    find_encoder()