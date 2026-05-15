import re

file_path = 'aeon_output/debug/ltxv_nodes_source.py'

with open(file_path, 'r') as f:
    content = f.read()

# Regex to find class definitions
class_pattern = re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.MULTILINE)
classes = class_pattern.findall(content)

print(f"Found {len(classes)} classes in {file_path}:")
for cls in sorted(classes):
    print(cls)

# Also search for any mentions of '4096' in the context of dimensions or projections
print("\n--- Searching for '4096' in source ---")
lines = content.splitlines()
for i, line in enumerate(lines):
    if '4096' in line:
        print(f"Line {i+1}: {line.strip()}")