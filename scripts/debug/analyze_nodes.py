import json
import os

def analyze_nodes():
    path = "scripts/debug/live_nodes.json"
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    with open(path, 'r') as f:
        try:
            nodes = json.load(f)
        except json.JSONDecodeError:
            print("Error decoding JSON")
            return

    print("--- LTX Related Nodes ---")
    ltx_nodes = []
    for node_name in nodes:
        if "LTX" in node_name.upper():
            ltx_nodes.append(node_name)
    
    for node in sorted(ltx_nodes):
        print(f"Found: {node}")

    print("\n--- Potential Loaders ---")
    loaders = [n for n in nodes if "Loader" in n]
    for l in sorted(loaders):
        if "LTX" in l.upper() or "Video" in l.upper():
            print(f"Loader: {l}")

    print("\n--- Potential Samplers ---")
    samplers = [n for n in nodes if "Sampler" in n]
    for s in sorted(samplers):
        if "LTX" in s.upper() or "Video" in s.upper():
            print(f"Sampler: {s}")

if __name__ == "__main__":
    analyze_nodes()