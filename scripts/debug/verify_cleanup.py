import os
import json
import fcntl
from aeon.main import register_models_for_agent, unregister_models_for_agent, MODEL_REGISTRY_PATH

def main():
    model_name = "Gemma-4-31B-Speculative-Q8_0"
    
    print(f"--- Initial Registry State ---")
    if os.path.exists(MODEL_REGISTRY_PATH):
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            print(f"Registry: {f.read()}")
    else:
        print("Registry does not exist.")

    print(f"\n--- Step 1: Registering {model_name} ---")
    # This should trigger _cleanup_stale_pids
    register_models_for_agent([model_name])
    
    print(f"\n--- Registry State after Registration ---")
    with open(MODEL_REGISTRY_PATH, 'r') as f:
        print(f"Registry: {f.read()}")

    print(f"\n--- Step 2: Unregistering {model_name} ---")
    # This should trigger _cleanup_stale_pids again and then remove our own PID
    unregister_models_for_agent([model_name])

    print(f"\n--- Final Registry State ---")
    if os.path.exists(MODEL_REGISTRY_PATH):
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            print(f"Registry: {f.read()}")
    else:
        print("Registry deleted (as expected if no users left).")

if __name__ == "__main__":
    main()