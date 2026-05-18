import json
import os

def scan_nodes():
    nodes_info_path = "aeon_output/debug/comfyui_nodes_info.json"
    output_path = "scripts/debug/ltxv_relevant_nodes.txt"
    
    if not os.path.exists(nodes_info_path):
        print(f"Error: {nodes_info_path} not found.")
        return

    print(f"Loading {nodes_info_path}...")
    with open(nodes_info_path, 'r') as f:
        try:
            nodes = json.load(f)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return

    keywords = ["LTX", "Embedding", "Connector", "Projection", "T5"]
    relevant_nodes = []

    for node_class, info in nodes.items():
        # Check class name
        if any(kw.lower() in node_class.lower() for kw in keywords):
            relevant_nodes.append((node_class, info))
            continue
        
        # Check inputs
        inputs = info.get("input", {})
        if isinstance(inputs, dict):
            # Check required/optional input names or tooltips
            input_str = str(inputs).lower()
            if any(kw.lower() in input_str for kw in keywords):
                relevant_nodes.append((node_class, info))
                continue
        
        # Check outputs
        outputs = info.get("output", {})
        if isinstance(outputs, dict):
            output_str = str(outputs).lower()
            if any(kw.lower() in output_str for kw in keywords):
                relevant_nodes.append((node_class, info))
                continue

    print(f"Found {len(relevant_nodes)} potentially relevant nodes.")
    
    with open(output_path, 'w') as f:
        for node_class, info in relevant_nodes:
            f.write(f"=== NODE: {node_class} ===\n")
            f.write(json.dumps(info, indent=2))
            f.write("\n\n" + "="*50 + "\n\n")

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    scan_nodes()