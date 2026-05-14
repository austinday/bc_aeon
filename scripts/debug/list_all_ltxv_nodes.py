import json
import os

def main():
    nodes_info_path = "aeon_output/debug/comfyui_nodes_info.json"
    if not os.path.exists(nodes_info_path):
        print(f"Error: {nodes_info_path} not found.")
        return

    with open(nodes_info_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return

    # The data is likely a list of nodes or a dict of nodes
    nodes = data if isinstance(data, list) else data.get("nodes", [])
    if not nodes:
        # Try to see if it's a dict where keys are class names
        if isinstance(data, dict):
            nodes = data.values()
        else:
            print("No nodes found in the JSON file.")
            return

    print(f"{'Class Type':<40} | {'Inputs':<50} | {'Outputs'}")
    print("-" * 110)

    found_ltxv = False
    for node in nodes:
        class_type = node.get("class_type", "Unknown")
        if "LTXV" in class_type.upper():
            found_ltxv = True
            inputs = node.get("inputs", {})
            outputs = node.get("outputs", {})
            
            # Format inputs and outputs for readability
            input_str = ", ".join(inputs.keys()) if isinstance(inputs, dict) else str(inputs)
            output_str = ", ".join(outputs.keys()) if isinstance(outputs, dict) else str(outputs)
            
            print(f"{class_type:<40} | {input_str[:48]:<50} | {output_str}")

    if not found_ltxv:
        print("No nodes containing 'LTXV' were found.")

if __name__ == "__main__":
    main()