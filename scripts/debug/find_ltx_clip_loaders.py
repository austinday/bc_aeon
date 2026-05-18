import json
import os

def find_ltx_clip_loaders():
    info_path = 'aeon_output/debug/comfyui_nodes_info.json'
    if not os.path.exists(info_path):
        print(f"Error: {info_path} not found.")
        return

    with open(info_path, 'r') as f:
        nodes = json.load(f)

    clip_loaders = []
    for node_name, info in nodes.items():
        # Check if it outputs CLIP
        outputs = info.get('output', [])
        if 'CLIP' in outputs:
            # Check if it's related to LTX, Gemma, or T5
            name_lower = node_name.lower()
            display_name = info.get('display_name', '').lower() if info.get('display_name') else ""
            description = info.get('description', '').lower() if info.get('description') else ""
            module = info.get('python_module', '').lower() if info.get('python_module') else ""
            
            if any(x in name_lower or x in display_name or x in description or x in module for x in ['ltx', 'gemma', 't5']):
                clip_loaders.append({
                    'name': node_name,
                    'display_name': info.get('display_name'),
                    'inputs': info.get('input', {}).get('required', {}),
                    'module': info.get('python_module'),
                    'description': info.get('description')
                })

    if not clip_loaders:
        print("No LTX-related CLIP loaders found.")
    else:
        print(f"Found {len(clip_loaders)} LTX-related CLIP loaders:\n")
        for loader in clip_loaders:
            print(f"--- {loader['name']} ---")
            print(f"Display Name: {loader['display_name']}")
            print(f"Module: {loader['module']}")
            print(f"Inputs: {json.dumps(loader['inputs'], indent=2)}")
            print(f"Description: {loader['description']}")
            print("\n")

if __name__ == "__main__":
    find_ltx_clip_loaders()