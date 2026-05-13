import json
import os

def inspect_nodes():
    info_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    target_nodes = ["LTXVImgToVideo", "LTXVImgToVideoInplace", "LTXVConcatAVLatent"]
    
    for node_name in target_nodes:
        if node_name in data:
            print(f"\n=== Node: {node_name} ===")
            node_info = data[node_name]
            print(f"Inputs: {json.dumps(node_info.get('inputs', {}), indent=2)}")
            print(f"Outputs: {json.dumps(node_info.get('outputs', {}), indent=2)}")
        else:
            print(f"\nNode {node_name} NOT found in info dump.")

if __name__ == "__main__":
    inspect_nodes()