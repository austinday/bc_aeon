import sys
import os

# Ensure the local directory is in the path
sys.path.append(os.getcwd())

try:
    from aeon.main import build_model_menu, LLAMACPP_MODELS
    print("Successfully imported aeon.main")
    
    # Simulate the local models list (empty for this test)
    local_models = []
    menu = build_model_menu(local_models)
    
    print("\n--- Generated Menu ---")
    selectable_count = 0
    for entry in menu:
        if entry.get('is_header'):
            print(f"[HEADER] {entry['label']}")
        else:
            selectable_count += 1
            print(f"{selectable_count:2}. {entry['label']}")
    
    # Check specifically for the NVFP4 model
    found = any('Gemma-4-31B-NVFP4' in entry.get('model', '') for entry in menu)
    print(f"\nModel 'Gemma-4-31B-NVFP4' found in menu: {found}")
    
    if not found:
        print("\nChecking LLAMACPP_MODELS list directly:")
        for m in LLAMACPP_MODELS:
            print(f" - {m['model']}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()