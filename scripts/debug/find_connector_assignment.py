import re

file_path = 'aeon_output/debug/ltxv_nodes_source.py'
with open(file_path, 'r') as f:
    content = f.read()

# Find all occurrences of video_embeddings_connector
pattern = re.compile(r'(\w+)\.video_embeddings_connector\s*=', re.MULTILINE)
matches = pattern.finditer(content)

for match in matches:
    start = match.start()
    # Look backwards to find the class definition
    prefix = content[:start]
    class_match = re.findall(r'class\s+(\w+)\(.*\):', prefix)
    if class_match:
        class_name = class_match[-1]
        print(f"Found assignment in class: {class_name}")
        print(f"Line content: {content[start:start+100].strip()}")
        
        # Extract the whole class to see the INPUT_TYPES
        class_start = content.rfind(f'class {class_name}', 0, start)
        # Find the end of the class (approximate by looking for the next class or end of file)
        next_class = content.find('class ', class_start + 1)
        class_body = content[class_start:next_class if next_class != -1 else len(content)]
        
        print(f"--- Class Body for {class_name} ---")
        print(class_body)
        print("--- End Class Body ---\n")

# Also search for any method that might be loading it
pattern_load = re.compile(r'load\s*\(.*embeddings_connectors.*', re.IGNORECASE)
matches_load = pattern_load.finditer(content)
for match in matches_load:
    print(f"Found potential load call: {content[match.start():match.start()+100].strip()}")