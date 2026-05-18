import json
import os

def analyze_history():
    file_path = "aeon_output/debug/comfyui_history.json"
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    print(f"Top-level type: {type(data)}")
    
    # If it's a dict, it's likely prompt_id -> run_data
    if isinstance(data, dict):
        print(f"Dict keys: {list(data.keys())[:10]}")
        for pid, run_data in data.items():
            if not isinstance(run_data, dict):
                continue
            
            # Look for I2V runs (they have a LoadImage node, usually node 10 in our tool)
            prompt = run_data.get('prompt', {})
            if isinstance(prompt, list):
                # Some ComfyUI versions return [id, prompt_id, prompt_dict]
                if len(prompt) >= 3 and isinstance(prompt[2], dict):
                    workflow = prompt[2]
                else:
                    continue
            elif isinstance(prompt, dict):
                workflow = prompt
            else:
                continue

            # Check if this is an I2V run (contains LoadImage)
            is_i2v = any(node.get('class_type') == 'LoadImage' for node in workflow.values() if isinstance(node, dict))
            if is_i2v:
                print(f"\n--- Found I2V Run (ID: {pid}) ---")
                # Inspect the latent nodes
                for node_id, node in workflow.items():
                    if isinstance(node, dict):
                        if node.get('class_type') == 'EmptyLTXVLatentVideo':
                            print(f"Node {node_id} (EmptyLTXVLatentVideo): {node.get('inputs')}")
                        if node.get('class_type') == 'LatentComposite':
                            print(f"Node {node_id} (LatentComposite): {node.get('inputs')}")
                
                # Check outputs
                outputs = run_data.get('outputs', {})
                print(f"Outputs: {outputs}")
                break
    
    # If it's a list, it's likely a sequence of runs
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        for i, run_data in enumerate(data):
            if not isinstance(run_data, dict):
                continue
            
            prompt = run_data.get('prompt', {})
            if isinstance(prompt, list):
                if len(prompt) >= 3 and isinstance(prompt[2], dict):
                    workflow = prompt[2]
                else:
                    continue
            elif isinstance(prompt, dict):
                workflow = prompt
            else:
                continue

            is_i2v = any(node.get('class_type') == 'LoadImage' for node in workflow.values() if isinstance(node, dict))
            if is_i2v:
                print(f"\n--- Found I2V Run (Index: {i}) ---")
                for node_id, node in workflow.items():
                    if isinstance(node, dict):
                        if node.get('class_type') == 'EmptyLTXVLatentVideo':
                            print(f"Node {node_id} (EmptyLTXVLatentVideo): {node.get('inputs')}")
                        if node.get('class_type') == 'LatentComposite':
                            print(f"Node {node_id} (LatentComposite): {node.get('inputs')}")
                
                outputs = run_data.get('outputs', {})
                print(f"Outputs: {outputs}")
                break
    else:
        print("Unknown top-level data type.")

if __name__ == "__main__":
    analyze_history()