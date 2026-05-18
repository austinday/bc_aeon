import re
import json

file_path = 'aeon_output/debug/ltxv_nodes_source.py'
with open(file_path, 'r') as f:
    content = f.read()

# Regex to find classes that look like ComfyUI nodes
# We look for 'class Name( ... ):' followed by 'INPUT_TYPES'
class_pattern = re.compile(r'class\s+(\w+)\s*\(.*?\):', re.MULTILINE)
input_types_pattern = re.compile(r'INPUT_TYPES\s*=\s*\{(.*?)\}', re.DOTALL)

nodes = {}
for match in class_pattern.finditer(content):
    class_name = match.group(1)
    start_pos = match.end()
    # Search for INPUT_TYPES within the next 2000 characters of the class definition
    class_body = content[start_pos:start_pos + 2000]
    input_match = input_types_pattern.search(class_body)
    if input_match:
        nodes[class_name] = input_match.group(1).strip()

print(json.dumps(nodes, indent=2))