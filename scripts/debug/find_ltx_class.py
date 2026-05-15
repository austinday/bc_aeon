import re

file_path = 'aeon_output/debug/ltxv_nodes_source.py'
with open(file_path, 'r') as f:
    content = f.read()

# Find the line with dim_threshold=4096
match = re.search(r'dim_threshold=4096', content)
if match:
    start_pos = match.start()
    # Look backwards for the nearest 'class ' definition
    prefix = content[:start_pos]
    class_matches = list(re.finditer(r'class\s+(\w+)', prefix))
    if class_matches:
        last_class = class_matches[-1]
        class_name = last_class.group(1)
        print(f"Found class: {class_name}")
        
        # Extract the class definition to see inputs/outputs
        # Find the end of the class (next class or end of file)
        end_pos = content.find('class ', start_pos)
        if end_pos == -1:
            end_pos = len(content)
        
        print("\n--- Class Definition ---")
        print(content[last_class.start():end_pos])
        print("--- End Class Definition ---")
    else:
        print("Found dim_threshold=4096 but no enclosing class.")
else:
    print("Could not find 'dim_threshold=4096' in the file.")