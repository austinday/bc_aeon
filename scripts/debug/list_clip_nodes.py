import json
import os

info_path = "aeon_output/debug/comfyui_nodes_info.json"
output_path = "aeon_output/debug/clip_nodes_list.txt"

if not os.path.exists(info_path):
    print(f"Error: {info_path} not found")
    exit(1)

with open(info_path, "r") as f:
    data = json.load(f)

clip_nodes = []
for node_name, node_info in data.items():
    outputs = node_info.get("output", [])
    if isinstance(outputs, list) and "CLIP" in outputs:
        clip_nodes.append(node_name)
    elif isinstance(outputs, str) and outputs == "CLIP":
        clip_nodes.append(node_name)

with open(output_path, "w") as f:
    f.write("\n".join(clip_nodes))

print(f"Found {len(clip_nodes)} nodes that return CLIP. List saved to {output_path}")
print("Nodes returning CLIP:")
for node in clip_nodes:
    print(node)