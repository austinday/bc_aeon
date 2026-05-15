import re

file_path = 'aeon_output/debug/ltxv_nodes_source.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find all class definitions
classes = re.findall(r'class\s+(\w+)\s*\(', content)
print(f"Found {len(classes)} classes in {file_path}")

loader_classes = [c for c in classes if 'Loader' in c]
print("\nClasses containing 'Loader':")
for lc in loader_classes:
    print(lc)

# Search for 4096 and 2048 and print the surrounding context
print("\nSearching for '4096' and '2048' context...")
lines = content.splitlines()
for i, line in enumerate(lines):
    if '4096' in line or '2048' in line:
        # Print the line and a few lines before/after to see the class/function
        start = max(0, i - 10)
        end = min(len(lines), i + 10)
        print(f"--- Match at line {i+1} ---")
        for j in range(start, end):
            prefix = ">> " if j == i else "   "
            print(f"{prefix}{lines[j]}")
        print("--------------------------\n")

# Specifically look for any class that might be the video loader
print("\nSearching for 'video' and 'encoder' in class names...")
video_encoder_classes = [c for c in classes if ('Video' in c or 'Text' in c) and 'Encoder' in c]
for vec in video_encoder_classes:
    print(vec)