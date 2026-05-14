import json
import os

nodes_info_path = "aeon_output/debug/comfyui_nodes_info.json"

if not os.path.exists(nodes_info_path):
    print(f"Error: {nodes_info_path} not found.")
    exit(1)

with open(nodes_info_path, "r") as f:
    nodes = json.load(f)

print(f"{'Node Name':<40} | {'Inputs':<50} | {'Outputs'}")
print("-" * 110)

for node_name, info in nodes.items():
    if "LTX" in node_name.upper():
        inputs = info.get("input", {})
        outputs = info.get("output", [])
        
        # Simplify inputs for display
        input_summary = ", ".join([f"{k}: {v[0]}" if isinstance(v, list) else f"{k}: {v}" for k, v in inputs.items()])
        output_summary = ", ".join(outputs)
        
        print(f"{node_name:<40} | {input_summary:<50} | {output_summary}")