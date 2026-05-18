import re
import os

file_path = 'aeon_output/debug/ltxv_nodes_source.py'
if not os.path.exists(file_path):
    print(f"File {file_path} not found.")
    exit(1)

with open(file_path, 'r') as f:
    content = f.read()

target_classes = ['LTXVTextProjection', 'LTXVConditioning']
results = {}

for cls_name in target_classes:
    # Find the class definition and its body
    pattern = rf'class {cls_name}\(.*\):(.*?(?=\nclass |\Z))'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        body = match.group(1)
        # Extract INPUTS
        inputs_match = re.search(r'INPUTS = (.*)', body)
        inputs = inputs_match.group(1).strip() if inputs_match else "Not found"
        
        # Extract RETURN_TYPES
        returns_match = re.search(r'RETURN_TYPES = (.*)', body)
        returns = returns_match.group(1).strip() if returns_match else "Not found"
        
        results[cls_name] = {"INPUTS": inputs, "RETURN_TYPES": returns}
    else:
        results[cls_name] = "Class not found"

for cls, info in results.items():
    print(f"--- {cls} ---")
    if isinstance(info, dict):
        print(f"INPUTS: {info['INPUTS']}")
        print(f"RETURN_TYPES: {info['RETURN_TYPES']}")
    else:
        print(info)