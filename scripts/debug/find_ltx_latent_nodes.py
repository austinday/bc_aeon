import json
import os

info_path = "aeon_output/debug/comfyui_nodes_info.json"

if not os.path.exists(info_path):
    print(f"Error: {info_path} not found.")
    exit(1)

with open(info_path, "r") as f:
    nodes_info = json.load(f)

print(f"Searching {len(nodes_info)} nodes for LTX nodes that output LATENT...\n")

found_nodes = []
for node_name, data in nodes_info.items():
    # Check if 'LTX' is in the name (case insensitive)
    if "LTX" in node_name.upper():
        # The 'output' field is usually a list of strings
        outputs = data.get("output", [])
        if isinstance(outputs, list):
            for out in outputs:
                if "LATENT" in str(out).upper():
                    found_nodes.append({
                        "node": node_name,
                        "outputs": outputs
                    })

if found_nodes:
    print(f"Found {len(found_nodes)} matching nodes:")
    for node in found_nodes:
        print(f"Node: {node['node']} | Outputs: {node['outputs']}")
else:
    print("No LTX nodes found that output LATENT.")

# Also search for ANY node that outputs LATENT and takes an IMAGE as input
print("\nSearching for ANY node that outputs LATENT and takes IMAGE as input...")
image_to_latent = []
for node_name, data in nodes_info.items():
    outputs = data.get("output", [])
    inputs = data.get("input", {}).get("required", {})
    
    has_latent_out = False
    if isinstance(outputs, list):
        for out in outputs:
            if "LATENT" in str(out).upper():
                has_latent_out = True
                break
    
    if has_latent_out:
        # Check if any required input is IMAGE
        for input_name, input_type in inputs.items():
            if isinstance(input_type, list) and "IMAGE" in str(input_type[0]).upper():
                image_to_latent.append({
                    "node": node_name,
                    "input": input_name,
                    "outputs": outputs
                })

if image_to_latent:
    print(f"Found {len(image_to_latent)} image-to-latent nodes:")
    for node in image_to_latent:
        print(f"Node: {node['node']} | Input: {node['input']} | Outputs: {node['outputs']}")
else:
    print("No image-to-latent nodes found.")