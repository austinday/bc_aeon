import re
import os

def extract_classes(source_file, target_file, class_names):
    if not os.path.exists(source_file):
        print(f"Source file {source_file} not found.")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    extracted_code = []
    for cls in class_names:
        # Regex to find the class and its body. 
        # This is a naive approach but works for most ComfyUI node files where classes are top-level.
        pattern = rf'(class {cls}.*?:.*?)(?=\nclass |\Z)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            extracted_code.append(f"# --- {cls} ---\n{match.group(1)}\n")
        else:
            extracted_code.append(f"# --- {cls} NOT FOUND ---")

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(extracted_code))
    print(f"Extracted classes to {target_file}")

if __name__ == "__main__":
    source = 'aeon_output/debug/ltxv_nodes_source.py'
    target = 'aeon_output/debug/extracted_ltx_nodes.py'
    classes = ['LTXVTextProjection', 'LTXVConditioning']
    extract_classes(source, target, classes)