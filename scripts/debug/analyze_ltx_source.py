import re
import os

FILE_PATH = "/home/aday/bc_aeon/aeon_output/debug/ltxv_nodes_source.py"

def analyze_source():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r') as f:
        content = f.read()

    # Find all class definitions
    classes = re.finditer(r'class\s+(\w+)\s*\(', content)
    
    ltx_classes = []
    for match in classes:
        class_name = match.group(1)
        if "LTX" in class_name:
            ltx_classes.append(class_name)

    print(f"Found {len(ltx_classes)} LTX-related classes: {ltx_classes}")

    # Extract the full body of interesting classes
    for class_name in ltx_classes:
        if "Loader" in class_name or "Encoder" in class_name or "Conditioning" in class_name:
            print(f"\n{'='*80}")
            print(f"ANALYZING CLASS: {class_name}")
            print(f"{'='*80}")
            
            # Find the start of the class
            start_idx = content.find(f"class {class_name}")
            # Find the end of the class (approximate by looking for the next class or end of file)
            # This is naive but usually works for these dumps
            next_class = content.find("class ", start_idx + 1)
            end_idx = next_class if next_class != -1 else len(content)
            
            class_body = content[start_idx:end_idx]
            print(class_body)

    # Search for '4096' in the whole file to see context
    print(f"\n{'='*80}")
    print("SEARCHING FOR '4096' CONTEXT")
    print(f"{'='*80}")
    for match in re.finditer(r'.*4096.*', content):
        print(match.group(0))

if __name__ == "__main__":
    analyze_source()