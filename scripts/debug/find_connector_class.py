import re
import os

file_path = 'aeon_output/debug/ltxv_nodes_source.py'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, 'r') as f:
    content = f.read()

# Find all class definitions
class_matches = re.finditer(r'class\s+(\w+)\s*\(', content)
classes = []
for match in class_matches:
    class_name = match.group(1)
    start_pos = match.start()
    # Find the end of the class by looking for the next 'class ' at the start of a line
    # This is a heuristic but usually works for ComfyUI nodes
    remaining = content[start_pos+1:]
    next_class = re.search(r'\nclass\s+', remaining)
    if next_class:
        end_pos = start_pos + 1 + next_class.start()
    else:
        end_pos = len(content)
    
    class_body = content[start_pos:end_pos]
    classes.append((class_name, class_body))

print(f"Found {len(classes)} classes. Searching for connector logic...\n")

target_keywords = ['embeddings_connectors', 'video_embeddings_connector', 'connector']

for name, body in classes:
    if any(kw in body for kw in target_keywords):
        print(f"--- Potential Match: {name} ---")
        # Print INPUT_TYPES if it exists
        input_match = re.search(r'INPUT_TYPES\s*=\s*(\{.*?\})', body, re.DOTALL)
        if input_match:
            print(f"INPUT_TYPES: {input_match.group(1)}")
        
        # Print lines that mention the keywords
        for line in body.split('\n'):
            if any(kw in line for kw in target_keywords):
                print(f"Line: {line.strip()}")
        print("\n")