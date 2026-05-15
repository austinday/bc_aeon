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

print(f"{'Class Name':<40} | {'Returns':<20} | {'Inputs'}")
print("-" * 80)

for match in class_matches:
    class_name = match.group(1)
    if 'LTX' in class_name and ('Loader' in class_name or 'Encoder' in class_name):
        # Find the RETURN_TYPES for this class
        # We look for the next occurrence of RETURN_TYPES within a reasonable distance
        start_pos = match.start()
        # Search for the end of the class or the start of another class
        end_pos = content.find('class ', start_pos + 1)
        if end_pos == -1:
            end_pos = len(content)
        
        class_body = content[start_pos:end_pos]
        
        return_match = re.search(r'RETURN_TYPES\s*=\s*([^,\n\r]+)', class_body)
        return_types = return_match.group(1).strip() if return_match else "Unknown"
        
        input_match = re.search(r'INPUT_TYPES\s*=\s*([^,\n\r]+)', class_body)
        input_types = input_match.group(1).strip() if input_match else "Unknown"
        
        print(f"{class_name:<40} | {return_types:<20} | {input_types}")

# Also search for any mention of 4096 in the context of linear layers or projections
print("\n--- Searching for 4096 projections ---")
projection_matches = re.finditer(r'nn\.Linear\(.*?, 4096\)', content)
for pm in projection_matches:
    # Find the class this is in
    pos = pm.start()
    # Look backwards for the nearest class definition
    search_back = content[:pos]
    classes_before = list(re.finditer(r'class\s+(\w+)\s*\(', search_back))
    if classes_before:
        last_class = classes_before[-1].group(1)
        print(f"Found 4096 projection in class: {last_class} at pos {pos}")