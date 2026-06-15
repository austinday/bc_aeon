import importlib
import inspect
import os
from pathlib import Path
from aeon.tools.loader import load_tools_from_directory
from aeon.tools.base import BaseTool

def diagnose():
    print("--- Starting Tool Loading Diagnosis ---")
    
    # Mock dependencies to satisfy loader
    deps = {
        'llm_client': None,
        'worker': None
    }
    
    print(f"Current Working Directory: {os.getcwd()}")
    
    # Try to load tools
    tools = load_tools_from_directory("aeon.tools", dependencies=deps, verbose=True)
    
    loaded_tool_names = [t.name for t in tools]
    print(f"\nSuccessfully loaded tools: {loaded_tool_names}")
    
    if 'expand_skills_category' not in loaded_tool_names:
        print("\n[!] CRITICAL: 'expand_skills_category' was NOT loaded.")
        
        # Deep dive into the module
        try:
            import aeon.tools.skills_manager_tool as smt
            print(f"Module aeon.tools.skills_manager_tool imported successfully.")
            
            for name, obj in inspect.getmembers(smt, inspect.isclass):
                if issubclass(obj, BaseTool):
                    print(f"Found BaseTool class: {name}")
                    sig = inspect.signature(obj.__init__)
                    print(f"  Signature: {sig}")
                    
                    # Test instantiation manually
                    try:
                        instance = obj(**deps)
                        print(f"  Manual instantiation SUCCESS: {instance.name}")
                    except Exception as e:
                        print(f"  Manual instantiation FAILED: {e}")
                else:
                    print(f"Class {name} is NOT a BaseTool subclass")
        except Exception as e:
            print(f"Error importing skills_manager_tool: {e}")

if __name__ == "__main__":
    diagnose()