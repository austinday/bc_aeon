import json
import os

history_path = "aeon_output/debug/comfyui_history.json"

if not os.path.exists(history_path):
    print(f"Error: {history_path} not found.")
    exit(1)

with open(history_path, 'r') as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        exit(1)

# The history is a dict where keys are prompt_ids
for prompt_id, history_item in data.items():
    # We are looking for the I2V run. 
    # I2V runs have a 'LoadImage' node (node 10 in our tool)
    prompt = history_item.get('prompt', {})
    if not prompt:
        continue
        
    # In the tool, I2V uses node 10 for LoadImage
    if "10" in prompt and prompt["10"].get("class_type") == "LoadImage":
        print(f"--- Found I2V Run (Prompt ID: {prompt_id}) ---")
        
        # Check the latent nodes
        for node_id in ["6", "11", "12"]:
            if node_id in prompt:
                print(f"Node {node_id} ({prompt[node_id].get('class_type')}): {json.dumps(prompt[node_id].get('inputs'), indent=2)}")
            else:
                print(f"Node {node_id} not found in prompt.")
        
        # Check the sampler
        if "7" in prompt:
            print(f"Node 7 (KSampler) inputs: {json.dumps(prompt['7'].get('inputs'), indent=2)}")
        
        # Check the video combine node
        if "9" in prompt:
            print(f"Node 9 (VHS_VideoCombine) inputs: {json.dumps(prompt['9'].get('inputs'), indent=2)}")
        
        print("-" * 40)

if not data:
    print("History file is empty.")