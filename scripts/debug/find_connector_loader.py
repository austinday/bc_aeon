import re

file_path = 'aeon_output/debug/ltxv_nodes_source.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find all class definitions
classes = re.finditer(r'class\s+(\w+)\s*\(', content)

for match in classes:
    class_name = match.group(1)
    # Find the end of the class (roughly, by looking for the next class or end of file)
    start_pos = match.start()
    # This is a simple heuristic; in a real parser we'd track indentation
    # But for a quick scan, we can just look at the block of text following the class definition
    
    # Search for keywords within the next 5000 characters of the class definition
    class_body = content[start_pos:start_pos + 5000]
    
    if 'embeddings' in class_body.lower() or 'connector' in class_body.lower():
        if 'load' in class_body.lower() or 'safetensors' in class_body.lower():
            print(f"Potential Connector Node Found: {class_name}")
            # Print the first few lines of the class to see the INPUT_TYPES
            lines = class_body.split('\n')
            for line in lines:
                if 'INPUT_TYPES' in line or 'return_types' in line:
                    print(f"  {line.strip()}")
            print("-" * 20)