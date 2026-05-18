import re

file_path = 'aeon_output/debug/ltxv_nodes_source.py'
with open(file_path, 'r') as f:
    content = f.read()

# Find all classes
classes = re.finditer(r'class\s+(\w+)\s*:\s*(?:[^:]*:\s*)?.*', content, re.DOTALL)

found_nodes = []

for match in classes:
    class_name = match.group(1)
    # Find the block of code for this class until the next class or end of file
    start_pos = match.start()
    # This is a naive way to find the end of the class, but we just need the INPUT_TYPES
    # We'll look for the next 'class ' at the start of a line
    remaining = content[start_pos:]
    class_body = ""
    lines = remaining.splitlines()
    for line in lines:
        if line.startswith('class ') and line != lines[0]:
            break
        class_body += line + '\n'
    
    if 'INPUT_TYPES' in class_body:
        # Check if 'connector' or 'embeddings' is in the INPUT_TYPES definition
        input_types_match = re.search(r'INPUT_TYPES\s*=\s*([^:]*:\s*\{.*?\}\s*,\s*[^)]*\))', class_body, re.DOTALL)
        if input_types_match:
            input_def = input_types_match.group(1)
            if 'connector' in input_def.lower() or 'embeddings' in input_def.lower():
                found_nodes.append({
                    'class': class_name,
                    'input_types': input_def
                })

if not found_nodes:
    # Fallback: just search for any mention of 'embeddings_connectors' in the class body
    for match in classes:
        class_name = match.group(1)
        start_pos = match.start()
        remaining = content[start_pos:]
        class_body = ""
        lines = remaining.splitlines()
        for line in lines:
            if line.startswith('class ') and line != lines[0]:
                break
            class_body += line + '\n'
        
        if 'embeddings_connectors' in class_body:
            found_nodes.append({
                'class': class_name,
                'input_types': 'Not found in INPUT_TYPES, but mentioned in body'
            })

for node in found_nodes:
    print(f"Found Node: {node['class']}")
    print(f"Input Types: {node['input_types']}")
    print("-" * 40)