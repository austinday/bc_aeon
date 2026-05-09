import json
import os

file_path = "aeon_output/debug/comfyui_history.json"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, 'r') as f:
    try:
        data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        exit(1)

print(f"Root type: {type(data)}")
if isinstance(data, dict):
    print(f"Root keys: {list(data.keys())}")
    for k, v in data.items():
        print(f"Key '{k}' type: {type(v)}")
        if isinstance(v, list) and len(v) > 0:
            print(f"  First element of '{k}' type: {type(v[0])}")
elif isinstance(data, list):
    print(f"Root length: {len(data)}")
    if len(data) > 0:
        print(f"First element type: {type(data[0])}")
        if isinstance(data[0], dict):
            print(f"  First element keys: {list(data[0].keys())}")