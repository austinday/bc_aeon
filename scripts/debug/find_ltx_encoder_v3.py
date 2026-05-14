import json
import os

def analyze_nodes():
    info_path = 'aeon_output/debug/comfyui_object_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        nodes = json.load(f)

    print(f"{'Node Name':<40} | {'Inputs':<60} | {'Outputs'}")
    print("-" * 110)

    # 1. Find STRING -> CONDITIONING
    print("\n--- STRING -> CONDITIONING Nodes ---")
    for name, data in nodes.items():
        inputs = data.get('input', {}).get('required', {})
        outputs = data.get('output', [])
        
        has_string_input = any('STRING' in str(val) for val in inputs.values())
        has_cond_output = 'CONDITIONING' in outputs
        
        if has_string_input and has_cond_output:
            input_str = ", ".join([f"{k}:{v}" for k, v in inputs.items()])
            output_str = ", ".join(outputs)
            print(f"{name:<40} | {input_str:<60} | {output_str}")

    # 2. Find all LTX nodes and their I/O
    print("\n--- All LTX-related Nodes ---")
    for name, data in nodes.items():
        if 'LTX' in name.upper():
            inputs = data.get('input', {}).get('required', {})
            outputs = data.get('output', [])
            
            input_str = ", ".join([f"{k}:{v}" for k, v in inputs.items()])
            output_str = ", ".join(outputs)
            print(f"{name:<40} | {input_str:<60} | {output_str}")

if __name__ == "__main__":
    analyze_nodes()