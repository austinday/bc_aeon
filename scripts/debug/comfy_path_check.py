import os
import sys

# Add ComfyUI to path
sys.path.append('/workspace/ComfyUI')

try:
    import folder_paths
    print("--- ComfyUI Path Diagnostics ---")
    
    # Check 'checkpoints'
    cp_paths = folder_paths.get_folder_paths('checkpoints')
    print(f"Checkpoints paths: {cp_paths}")
    if cp_paths:
        for p in cp_paths:
            if os.path.exists(p):
                print(f"Files in {p}: {os.listdir(p)}")
            else:
                print(f"Path {p} does not exist")
    
    # Check 'text_encoders'
    te_paths = folder_paths.get_folder_paths('text_encoders')
    print(f"Text Encoders paths: {te_paths}")
    if te_paths:
        for p in te_paths:
            if os.path.exists(p):
                print(f"Files in {p}: {os.listdir(p)}")
            else:
                print(f"Path {p} does not exist")
                
    # Check what ComfyUI actually lists as available files
    print(f"ComfyUI detected checkpoints: {folder_paths.get_filename_list('checkpoints')}")
    print(f"ComfyUI detected text_encoders: {folder_paths.get_filename_list('text_encoders')}")
    
except Exception as e:
    print(f"Error during diagnostics: {e}")
    import traceback
    traceback.print_exc()