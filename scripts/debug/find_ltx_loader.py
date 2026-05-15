import json
import os

def analyze_nodes():
    info_path = "aeon_output/debug/comfyui_nodes_info.json"
    source_path = "aeon_output/debug/ltxv_nodes_source.py"
    
    print("--- Analyzing comfyui_nodes_info.json ---")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            try:
                data = json.load(f)
                ltx_loaders = []
                for node_name, info in data.items():
                    if "LTX" in node_name.upper() and "LOADER" in node_name.upper():
                        ltx_loaders.append(node_name)
                
                print(f"Found {len(ltx_loaders)} LTX loaders:")
                for loader in ltx_loaders:
                    print(f"Node: {loader}")
                    print(f"  Inputs: {info.get('input', {}).get('required', 'N/A')}")
                    print(f"  Outputs: {info.get('output', 'N/A')}")
                    print("-" * 20)
            except Exception as e:
                print(f"Error parsing JSON: {e}")
    else:
        print("info_path not found.")

    print("\n--- Analyzing ltxv_nodes_source.py for T5/4096 ---")
    if os.path.exists(source_path):
        with open(source_path, 'r') as f:
            content = f.read()
            lines = content.splitlines()
            
            # Find classes that mention T5 or 4096
            current_class = None
            for i, line in enumerate(lines):
                if line.strip().startswith("class "):
                    current_class = line.strip().split(" ")[1].split("(")[0]
                
                if current_class and ("T5" in line or "4096" in line):
                    print(f"Line {i+1} in class {current_class}: {line.strip()}")
    else:
        print("source_path not found.")

if __name__ == "__main__":
    analyze_nodes()