import json
import os

def inspect_nodes(nodes_file, target_nodes):
    if not os.path.exists(nodes_file):
        print(f"File {nodes_file} not found.")
        return

    with open(nodes_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from {nodes_file}")
            return

    for node in target_nodes:
        if node in data:
            print(f"--- Node: {node} ---")
            print(json.dumps(data[node], indent=2))
            print("\n")
        else:
            print(f"Node {node} not found in {nodes_file}")

target_nodes = [
    "LTXAVTextEncoderLoader",
    "LTXVConditioning",
    "TextGenerateLTX2Prompt",
    "CLIPTextEncode",
    "CLIPLoader"
]

inspect_nodes("aeon_output/debug/comfyui_nodes_info.json", target_nodes)
inspect_nodes("aeon_output/debug/ltx_nodes_detailed.json", target_nodes)