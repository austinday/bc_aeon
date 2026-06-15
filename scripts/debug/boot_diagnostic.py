import sys
import os
from pathlib import Path

# Ensure the current directory is in path so we can import aeon
sys.path.insert(0, os.getcwd())

try:
    from aeon.tools.loader import load_tools_from_directory
    from aeon.tools.base import BaseTool
    print("[DIAGNOSTIC] Imports successful.")
except Exception as e:
    print(f"[DIAGNOSTIC] Import failed: {e}")
    sys.exit(1)

def run_diagnostic():
    print("[DIAGNOSTIC] Simulating tool loading...")
    
    # Mock dependencies as the main app would
    deps = {} 
    
    # Load tools from the package
    tools = load_tools_from_directory("aeon.tools", dependencies=deps, verbose=True)
    
    print("\n--- LOADED TOOLS LIST ---")
    tool_names = [t.name for t in tools]
    for name in tool_names:
        print(f"Found: {name}")
    print("--------------------------\n")
    
    target = "expand_skills_category"
    if target in tool_names:
        print(f"[RESULT] SUCCESS: '{target}' was found and loaded.")
    else:
        print(f"[RESULT] FAILURE: '{target}' was NOT found in the loaded tools list.")
        
        # Deep dive into the module to see if the class exists but was skipped
        try:
            import aeon.tools.skills_manager_tool as smt
            classes = [name for name, obj in vars(smt).items() if isinstance(obj, type) and issubclass(obj, BaseTool)]
            print(f"[DEBUG] Classes in skills_manager_tool that are BaseTools: {classes}")
        except Exception as e:
            print(f"[DEBUG] Could not inspect module: {e}")

if __name__ == "__main__":
    run_diagnostic()