import json
import os

def search_nodes():
    info_path = 'aeon_output/debug/comfyui_object_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        nodes = json.load(f)

    print(f"{'Node Name':<40} | {'Inputs':<50} | {'Outputs'}")
    print("-" * 110)

    found_any = False
    for node_name, data in nodes.items():
        # Check if it outputs CONDITIONING
        outputs = data.get('output', [])
        if 'CONDITIONING' not in outputs:
            continue

        # Search for T5 or LTX in name, inputs, or tooltips
        search_str = (node_name + str(data.get('input', {}))).lower()
        
        # Also check tooltips specifically
        tooltips = []
        inputs = data.get('input', {}).get('required', {})
        for inp_name, inp_data in inputs.items():
            if isinstance(inp_data, list) and len(inp_data) > 1:
                tooltip = inp_data[1].get('tooltip', '') if isinstance(inp_data[1], dict) else ''
                tooltips.append(tooltip)
        
        full_search_text = search_str + " ".join(tooltips).lower()

        if 't5' in full_search_text or 'ltx' in full_search_text:
            found_any = True
            # Format inputs for display
            req_inputs = inputs.keys()
            input_str = ", ".join(req_inputs)
            print(f"{node_name:<40} | {input_str:<50} | {', '.join(outputs)}")

    if not found_any:
        print("No nodes found matching 'T5' or 'LTX' that output CONDITIONING.")

    print("\n\n--- All CLIP -> CONDITIONING Nodes ---")
    print(f"{'Node Name':<40} | {'Inputs':<50} | {'Outputs'}")
    print("-" * 110)
    
    clip_to_cond_found = False
    for node_name, data in nodes.items():
        outputs = data.get('output', [])
        if 'CONDITIONING' not in outputs:
            continue
        
        inputs = data.get('input', {}).get('required', {})
        # Check if any input is 'CLIP'
        # Note: The object info might store the type in the value list
        is_clip_input = False
        for inp_name, inp_val in inputs.items():
            if isinstance(inp_val, list) and 'CLIP' in inp_val:
                is_clip_input = True
                break
            if inp_val == 'CLIP':
                is_clip_input = True
                break
        
        if is_clip_input:
            clip_to_cond_found = True
            input_str = ", ".join(inputs.keys())
            print(f"{node_name:<40} | {input_str:<50} | {', '.join(outputs)}")

    if not clip_to_cond_found:
        print("No nodes found that take CLIP and output CONDITIONING.")

if __name__ == "__main__":
    search_nodes()