import json
import os

def inspect_nodes():
    info_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return

    target_nodes = [
        "LTXVImgToVideo", 
        "LTXVImgToVideoInplace", 
        "LTXVConcatAVLatent", 
        "LTXVEnhanceAVideoKJ"
    ]

    # The data is likely a dictionary where keys are node names or it's a list of nodes.
    # Let's check the structure.
    if isinstance(data, list):
        nodes_dict = {node.get('class_type'): node for node in data if 'class_type' in node}
    elif isinstance(data, dict):
        # If it's a dict, it might be { "node_name": { "inputs": ..., "outputs": ... } }
        nodes_dict = data
    else:
        print(f"Unexpected data type: {type(data)}")
        return

    for target in target_nodes:
        print(f"\n=== Node: {target} ===")
        if target in nodes_dict:
            node_info = nodes_dict[target]
            # ComfyUI /object_info usually has 'inputs' and 'outputs' keys
            inputs = node_info.get('inputs', 'No inputs found')
            outputs = node_info.get('outputs', 'No outputs found')
            print(f"Inputs: {json.dumps(inputs, indent=2)}")
            print(f"Outputs: {json.dumps(outputs, indent=2)}")
        else:
            print(f"Node {target} not found in the dump.")
            # Try a partial match in case the name is slightly different
            matches = [k for k in nodes_dict.keys() if target in k]
            if matches:
                print(f"Possible matches: {matches}")

if __name__ == "__main__":
    inspect_nodes()