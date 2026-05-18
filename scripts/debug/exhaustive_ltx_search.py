import re
import json

SOURCE_FILE = "aeon_output/debug/ltxv_nodes_source.py"
INFO_FILE = "aeon_output/debug/comfyui_nodes_info.json"

def analyze_source():
    print("--- Analyzing Source Code ---")
    with open(SOURCE_FILE, "r") as f:
        content = f.read()

    # Find all class definitions
    classes = re.findall(r"class\s+(\w+)\s*\(", content)
    print(f"Found {len(classes)} classes in {SOURCE_FILE}")
    
    # Filter for LTX related classes
    ltx_classes = [c for c in classes if "LTX" in c]
    print(f"Found {len(ltx_classes)} LTX-related classes")
    
    for cls in ltx_classes:
        # Try to find the define_schema method for this class
        pattern = rf"class\s+{cls}.*?def\s+define_schema\s*\(cls\):.*?return\s+io\.Schema\((.*?)\),"
        # This regex is a bit simplistic, let's just find the block
        start_idx = content.find(f"class {cls}")
        end_idx = content.find("class ", start_idx + 1)
        if end_idx == -1: end_idx = len(content)
        
        class_body = content[start_idx:end_idx]
        if "define_schema" in class_body:
            # Extract the display_name or node_id
            name_match = re.search(r'display_name="([^"]+)"', class_body)
            id_match = re.search(r'node_id="([^"]+)"', class_body)
            display_name = name_match.group(1) if name_match else "Unknown"
            node_id = id_match.group(1) if id_match else "Unknown"
            print(f"Class: {cls} | ID: {node_id} | Display: {display_name}")

def analyze_info():
    print("\n--- Analyzing Nodes Info JSON ---")
    try:
        with open(INFO_FILE, "r") as f:
            data = json.load(f)
        
        for node_name, info in data.items():
            # Check if node name or any info contains LTX and (Video or Encoder)
            node_str = json.dumps(info).lower()
            if "ltx" in node_name.lower() and ("video" in node_str or "encoder" in node_str or "text" in node_str):
                print(f"Node: {node_name} | Info: {info}")
    except Exception as e:
        print(f"Error analyzing info file: {e}")

if __name__ == "__main__":
    analyze_source()
    analyze_info()