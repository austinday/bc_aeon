import re

file_path = 'aeon_output/debug/ltxv_nodes_source.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find all classes that look like loaders
loader_classes = re.findall(r'class (\w+Loader\w*):', content)
print(f"Found loader classes: {loader_classes}")

for cls in loader_classes:
    print(f"\n--- Analyzing class {cls} ---")
    # Extract the class body
    start_idx = content.find(f'class {cls}:')
    # Find the end of the class by looking for the next class definition at the same indentation level
    # This is a naive approach, but should work for simple class structures
    remaining = content[start_idx + len(cls) + 1:]
    
    # Look for the next 'class ' at the start of a line
    match = re.search(r'\nclass ', remaining)
    if match:
        class_body = remaining[:match.start()]
    else:
        class_body = remaining
    
    # Search for INPUT_TYPES
    input_match = re.search(r'INPUT_TYPES = (\{.*?\})', class_body, re.DOTALL)
    if input_match:
        print(f"INPUT_TYPES for {cls}:")
        print(input_match.group(1))
    else:
        print(f"No INPUT_TYPES found for {cls}")

    # Search for return type in the return method
    return_match = re.search(r'def .*?\(self, .*?\):.*?return (.*?)\n', class_body, re.DOTALL)
    if return_match:
        print(f"Return statement for {cls}: {return_match.group(1)}")

# Also search for any mention of '4096' or '2048' in the context of embeddings or projections
print("\n--- Searching for embedding dimensions ---")
for i, line in enumerate(content.splitlines()):
    if '4096' in line or '2048' in line:
        if 'embed' in line.lower() or 'proj' in line.lower() or 'dim' in line.lower():
            print(f"Line {i+1}: {line.strip()}")