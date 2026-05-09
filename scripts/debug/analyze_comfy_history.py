import json
import os

history_path = 'aeon_output/debug/comfyui_history.json'

if not os.path.exists(history_path):
    print(f"Error: History file not found at {history_path}")
    exit(1)

with open(history_path, 'r') as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        exit(1)

# The history is usually a dict where keys are prompt_ids
for prompt_id, run_data in data.items():
    prompt = run_data.get('prompt', {})
    if not prompt:
        continue
    
    # Check if this is an I2V run (contains LoadImage)
    is_i2v = any('LoadImage' in str(node.get('class_type', '')) for node in prompt.values())
    
    if is_i2v:
        print(f"\n{'='*60}")
        print(f"Analyzing I2V Run: {prompt_id}")
        print(f"{'='*60}")
        
        # Look for the EmptyLTXVLatentVideo node
        for node_id, node in prompt.items():
            if node.get('class_type') == 'EmptyLTXVLatentVideo':
                print(f"Node {node_id} (EmptyLTXVLatentVideo) inputs: {node.get('inputs')}")
        
        # Look for the KSampler node
        for node_id, node in prompt.items():
            if node.get('class_type') == 'KSampler':
                print(f"Node {node_id} (KSampler) inputs: {node.get('inputs')}")
        
        # Look for the LatentComposite node
        for node_id, node in prompt.items():
            if node.get('class_type') == 'LatentComposite':
                print(f"Node {node_id} (LatentComposite) inputs: {node.get('inputs')}")

        # Check outputs
        outputs = run_data.get('outputs', {})
        print(f"Outputs: {json.dumps(outputs, indent=2)}")

if not data:
    print("No data found in history file.")