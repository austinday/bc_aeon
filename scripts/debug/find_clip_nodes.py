import re
import os

file_path = '/home/aday/bc_aeon/aeon_output/debug/ltxv_nodes_source.py'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, 'r') as f:
    content = f.read()

# Regex to find class definitions and their RETURN_TYPES
# This looks for 'class ClassName' followed by 'RETURN_TYPES = ...'
class_pattern = re.compile(r'class\s+(\w+)\s*:\s*.*?\n\s*RETURN_TYPES\s*=\s*([^,\n\r\s]+)', re.DOTALL)

matches = class_pattern.findall(content)

print(f"Found {len(matches)} classes with RETURN_TYPES defined.")
for class_name, return_type in matches:
    if 'CLIP' in return_type:
        print(f"Class: {class_name} | Returns: {return_type}")

# Also search for any class that mentions 'CLIP' in its return types even if it's a tuple
all_classes = re.findall(r'class\s+(\w+)\s*:', content)
for cls in all_classes:
    # Find the block for this class
    start_idx = content.find(f'class {cls}')
    # Find the next class or end of file
    end_idx = content.find('class ', start_idx + 1)
    if end_idx == -1:
        end_idx = len(content)
    
    class_body = content[start_idx:end_idx]
    if 'RETURN_TYPES' in class_body and 'CLIP' in class_body:
        # Extract the actual RETURN_TYPES line
        rt_match = re.search(r'RETURN_TYPES\s*=\s*([^,\n\r]+)', class_body)
        if rt_match:
            print(f"Found CLIP node: {cls} | RETURN_TYPES = {rt_match.group(1)}")