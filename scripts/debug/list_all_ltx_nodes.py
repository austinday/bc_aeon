import json
import os

def analyze_nodes():
    info_path = 'aeon_output/debug/comfyui_nodes_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return

    print(f"{'Node Class':<40} | {'Inputs':<50} | {'Outputs':<30}")
    print("-" * 125)

    # The data structure is usually a dict where keys are node classes
    for node_class, info in data.items():
        if 'LTX' in node_class.upper():
            # Handle different possible JSON structures
            inputs = info.get('input', {})
            outputs = info.get('output', {})
            
            # Simplify inputs for printing
            input_str = str(inputs) if inputs else "None"
            output_str = str(outputs) if outputs else "None"
            
            print(f"{node_class:<40} | {input_str[:47]+'...' if len(input_str)>47 else input_str:<50} | {output_str[:27]+'...' if len(output_str)>27 else output_str:<30}")

if __name__ == "__main__":
    analyze_nodes()