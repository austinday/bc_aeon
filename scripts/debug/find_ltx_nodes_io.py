import json
import os

def analyze_ltx_nodes():
    info_path = 'aeon_output/debug/comfyui_object_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    print(f"{'Node Name':<40} | {'Inputs':<50} | {'Outputs':<20}")
    print("-" * 115)

    for node_name, info in data.items():
        if 'LTX' in node_name.upper():
            inputs = []
            if 'input' in info:
                req = info['input'].get('required', {})
                for k, v in req.items():
                    # The value is usually a list where the first element is the type
                    type_val = v[0] if isinstance(v, list) else v
                    inputs.append(f"{k}:{type_val}")
            
            outputs = []
            if 'output' in info:
                for out in info['output']:
                    outputs.append(out)
            
            print(f"{node_name:<40} | {', '.join(inputs):<50} | {', '.join(outputs):<20}")

if __name__ == "__main__":
    analyze_ltx_nodes()