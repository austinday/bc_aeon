import json
import os

def extract_details(info_path, target_nodes):
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    results = {}
    for node_name in target_nodes:
        if node_name in data:
            results[node_name] = data[node_name]
        else:
            # Try case-insensitive or partial match
            found = False
            for k, v in data.items():
                if node_name.lower() in k.lower():
                    results[k] = v
                    found = True
                    break
            if not found:
                print(f"Warning: Node {node_name} not found in info file.")

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    info_file = "aeon_output/debug/comfyui_nodes_info.json"
    nodes_to_check = [
        "LTXVConcatAVLatent",
        "LTXVImgToVideo",
        "LTXVImgToVideoInplace",
        "LTXVImgToVideoInplaceKJ",
        "LTXVEnhanceAVideoKJ",
        "EmptyLTXVLatentVideo",
        "ModelSamplingLTXV"
    ]
    extract_details(info_file, nodes_to_check)