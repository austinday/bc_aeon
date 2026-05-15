import re
import os

file_path = 'aeon_output/debug/ltxv_nodes_source.py'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, 'r') as f:
    content = f.read()

# Regex to find classes and their INPUT_TYPES
# This looks for 'class ClassName' followed by 'INPUT_TYPES = {'
class_pattern = re.compile(r'class\s+(\w+)\s*:\s*.*?\n\s*INPUT_TYPES\s*=\s*(\{.*?\})', re.DOTALL)

matches = class_pattern.findall(content)

if not matches:
    print("No classes with INPUT_TYPES found using the primary regex. Trying a broader search...")
    # Broader search: find all classes, then look for INPUT_TYPES in the following lines
    all_classes = re.findall(r'class\s+(\w+)\s*:', content)
    for cls in all_classes:
        # Find the block of code for this class (until the next class or end of file)
        start_idx = content.find(f'class {cls}')
        end_idx = content.find('class ', start_idx + 1)
        if end_idx == -1:
            end_idx = len(content)
        
        class_body = content[start_idx:end_idx]
        input_match = re.search(r'INPUT_TYPES\s*=\s*(\{.*?\})', class_body, re.DOTALL)
        if input_match:
            print(f"Found Class: {cls}")
            print(f"INPUT_TYPES: {input_match.group(1)}")
            print("-" * 40)
else:
    for cls, inputs in matches:
        print(f"Found Class: {cls}")
        print(f"INPUT_TYPES: {inputs}")
        print("-" * 40)

# Also search for any mention of 'embeddings_connectors' in the file to see where it's used
print("\nSearching for 'embeddings_connectors' in source...")
for i, line in enumerate(content.splitlines()):
    if 'embeddings_connectors' in line:
        print(f"Line {i+1}: {line.strip()}")