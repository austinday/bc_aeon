import re
import os

def parse_ltx_source(file_path, output_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find classes and their associated RETURN_TYPES and INPUT_TYPES
    # This looks for 'class ClassName' and then the following RETURN_TYPES/INPUT_TYPES assignments
    class_pattern = re.compile(r'class\s+(\w+)\s*:\s*.*?(?=class\s+\w+\s*:|$)', re.DOTALL)
    
    results = []
    for match in class_pattern.finditer(content):
        class_name = match.group(1)
        class_body = match.group(0)
        
        return_types = re.search(r'RETURN_TYPES\s*=\s*([^ \n\r]+)', class_body)
        input_types = re.search(r'INPUT_TYPES\s*=\s*([^ \n\r]+)', class_body)
        
        ret = return_types.group(1) if return_types else "Not Found"
        inp = input_types.group(1) if input_types else "Not Found"
        
        results.append(f"Class: {class_name}\n  RETURN_TYPES: {ret}\n  INPUT_TYPES: {inp}\n{'-'*40}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    
    print(f"Successfully parsed {len(results)} classes to {output_path}")

if __name__ == "__main__":
    source_file = "/home/aday/bc_aeon/aeon_output/debug/ltxv_nodes_source.py"
    output_file = "/home/aday/bc_aeon/aeon_output/debug/ltx_node_map.txt"
    parse_ltx_source(source_file, output_file)