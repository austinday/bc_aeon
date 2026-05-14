import json
import os

def find_conditioning_nodes():
    info_path = 'aeon_output/debug/comfyui_object_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        data = json.load(f)

    conditioning_nodes = []
    for node_name, info in data.items():
        outputs = info.get('output', [])
        if 'CONDITIONING' in outputs:
            inputs = info.get('input', {}).get('required', {})
            conditioning_nodes.append({
                'name': node_name,
                'inputs': inputs,
                'outputs': outputs
            })

    output_file = 'aeon_output/debug/all_conditioning_nodes.txt'
    with open(output_file, 'w') as f:
        for node in conditioning_nodes:
            f.write(f"Node: {node['name']}\n")
            f.write(f"Inputs: {json.dumps(node['inputs'], indent=2)}\n")
            f.write(f"Outputs: {node['outputs']}\n")
            f.write("-" * 40 + "\n")

    print(f"Found {len(conditioning_nodes)} nodes outputting CONDITIONING. Saved to {output_file}")

if __name__ == "__main__":
    find_conditioning_nodes()