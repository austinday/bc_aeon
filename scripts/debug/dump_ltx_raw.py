import json
import os

def dump_nodes():
    json_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return

    target_nodes = [
        "LTXVImgToVideo", 
        "LTXVImgToVideoInplace", 
        "LTXVConcatAVLatent", 
        "LTXVEnhanceAVideoKJ",
        "LTXVImgToVideoInplaceKJ"
    ]

    for node in target_nodes:
        print(f"\n{'='*20} {node} {'='*20}")
        if node in data:
            print(json.dumps(data[node], indent=2))
        else:
            print(f"Node {node} not found in the dump.")

if __name__ == "__main__":
    dump_nodes()