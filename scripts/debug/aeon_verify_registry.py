import os
import json
from aeon.main import register_models_for_agent, unregister_models_for_agent, MODEL_REGISTRY_PATH

def main():
    model_name = "Gemma-4-31B-MTP-Q8_0"
    
    # 1. Check current registry
    if os.path.exists(MODEL_REGISTRY_PATH):
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            registry = json.load(f)
            pids = registry.get(model_name, [])
            print(f"[INFO] Current PIDs for {model_name}: {pids}")
    else:
        print("[INFO] Registry file not found.")
        return

    # 2. Register this script's PID so that unregister_models_for_agent 
    # actually enters the logic for this model.
    print(f"\n--- Registering this script (PID {os.getpid()}) for {model_name} ---")
    register_models_for_agent([model_name])

    # 3. Now unregister to trigger the diagnostic printout of remaining users
    print(f"\n--- Triggering unregister_models_for_agent ---")
    unregister_models_for_agent([model_name])
    print(f"--- End of unregistration trigger ---\n")

if __name__ == "__main__":
    main()