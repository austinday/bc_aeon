import json
import os

json_path = 'aeon_output/debug/ltx_nodes_detailed.json'

if not os.path.exists(json_path):
    print(f"File not found: {json_path}")
    exit(1)

with open(json_path, 'r') as f:
    data = json.load(f)

target_nodes = ['LTXVTextProjection', 'LTXVConditioning', 'LTXVImgToVideo', 'EmptyLTXVLatentVideo']

for node in target_nodes:
    if node in data:
        print(f"--- {node} ---")
        print(f"Inputs: {json.dumps(data[node].get('input'), indent=2)}")
        print(f"Outputs: {json.dumps(data[node].get('output'), indent=2)}")
        print("\n")
    else:
        print(f"--- {node} NOT FOUND ---")