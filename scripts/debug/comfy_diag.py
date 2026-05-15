import os
from ComfyUI import folder_paths

def diagnose():
    print("=== ComfyUI Model Diagnosis ===")
    
    targets = ['checkpoints', 'text_encoders', 'vae', 'unet']
    
    for target in targets:
        print(f"\n--- Target: {target} ---")
        try:
            path = folder_paths.get_folder_paths(target)
            print(f"Configured Path: {path}")
            
            if isinstance(path, list):
                for p in path:
                    print(f"  Checking directory: {p}")
                    if os.path.exists(p):
                        print(f"    Exists: Yes")
                        print(f"    Contents: {os.listdir(p)}")
                    else:
                        print(f"    Exists: No")
            else:
                print(f"  Checking directory: {path}")
                if os.path.exists(path):
                    print(f"    Exists: Yes")
                    print(f"    Contents: {os.listdir(path)}")
                else:
                    print(f"    Exists: No")
            
            files = folder_paths.get_filename_list(target)
            print(f"ComfyUI Detected Files: {files}")
            
        except Exception as e:
            print(f"Error diagnosing {target}: {e}")

    print("\n=== Physical Root Check ===")
    root_models = "/workspace/ComfyUI/models"
    if os.path.exists(root_models):
        print(f"Root exists. Contents: {os.listdir(root_models)}")
    else:
        print("Root models directory NOT FOUND")

if __name__ == "__main__":
    diagnose()