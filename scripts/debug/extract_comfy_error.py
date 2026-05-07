import json
import os

def extract_error():
    file_path = 'aeon_output/debug/comfyui_history.json'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print("--- ComfyUI History Analysis ---")
        
        # The file contains the data for a single prompt (history_data)
        status = data.get('status', {})
        print(f"Status: {status}")
        
        messages = status.get('messages', [])
        if messages:
            print("\nMessages:")
            for msg in messages:
                print(f"- {msg}")
        else:
            print("\nNo status messages found.")

        outputs = data.get('outputs', {})
        print(f"\nOutputs: {outputs}")
        
        if not outputs:
            print("\nConclusion: No outputs were produced. The workflow likely failed during execution.")
        else:
            print("\nConclusion: Outputs were produced, but the expected node (9) might be missing.")

    except Exception as e:
        print(f"An error occurred while parsing the JSON: {e}")

if __name__ == "__main__":
    extract_error()