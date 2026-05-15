import json
import os

files_to_check = [
    "aeon_output/debug/comfyui_nodes_info.json",
    "aeon_output/debug/ltx_nodes_detailed.json"
]

keywords = ["LTX", "Sampler", "Encoder", "Prompt", "Conditioning"]

for file_path in files_to_check:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    print(f"Searching in {file_path}...")
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
            
    matches = []
    for node_name, info in data.items():
        # Check if node name contains LTX or any of the keywords
        if any(kw.lower() in node_name.lower() for kw in keywords):
            matches.append(node_name)
            
    print(f"Found {len(matches)} potential nodes in {file_path}:")
    for m in matches:
        print(f"- {m}")
    print("-" * 20)