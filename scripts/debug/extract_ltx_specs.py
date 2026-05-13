import json
import os

def extract_specs():
    json_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        try:
            nodes = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return

    targets = [
        "LTXVImgToVideo", 
        "LTXVImgToVideoInplace", 
        "LTXVConcatAVLatent", 
        "LTXVEnhanceAVideoKJ"
    ]

    for node_name, node_info in nodes.items():
        if any(target in node_name for target in targets):
            print(f"\n{'='*20} {node_name} {'='*20}")
            
            # Extract Inputs
            print("[INPUTS]")
            inputs = node_info.get("inputs", {})
            if isinstance(inputs, list):
                # ComfyUI /object_info format: [ [type, {name, type, ...}], ... ]
                for i, input_item in enumerate(inputs):
                    if isinstance(input_item, list) and len(input_item) > 1:
                        meta = input_item[1]
                        name = meta.get("name", "unknown")
                        type_ = meta.get("type", "unknown")
                        print(f"  {i}: {name} (Type: {type_})")
                    else:
                        print(f"  {i}: Unexpected input format: {input_item}")
            elif isinstance(inputs, dict):
                for name, details in inputs.items():
                    print(f"  {name}: {details}")
            else:
                print("  No inputs found or unsupported format.")

            # Extract Outputs
            print("[OUTPUTS]")
            outputs = node_info.get("outputs", {})
            if isinstance(outputs, list):
                for i, output_item in enumerate(outputs):
                    if isinstance(output_item, list) and len(output_item) > 1:
                        meta = output_item[1]
                        name = meta.get("name", "unknown")
                        type_ = meta.get("type", "unknown")
                        print(f"  {i}: {name} (Type: {type_})")
                    else:
                        print(f"  {i}: Unexpected output format: {output_item}")
            elif isinstance(outputs, dict):
                for name, details in outputs.items():
                    print(f"  {name}: {details}")
            else:
                print("  No outputs found or unsupported format.")

if __name__ == "__main__":
    extract_specs()