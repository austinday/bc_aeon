import importlib
import inspect
import sys
from aeon.tools.base import BaseTool

def diagnose():
    module_name = "aeon.tools.generate_video"
    print(f"--- Diagnosing {module_name} ---")
    
    try:
        print(f"Attempting to import {module_name}...")
        module = importlib.import_module(module_name)
        print("Import successful.")
    except Exception as e:
        print(f"Import FAILED: {e}")
        return

    print("\nSearching for classes in module...")
    classes = inspect.getmembers(module, inspect.isclass)
    print(f"Found {len(classes)} classes.")
    
    for name, obj in classes:
        print(f"\nChecking class: {name}")
        
        # Check 1: Module match
        print(f"  - obj.__module__: {obj.__module__}")
        print(f"  - expected module: {module_name}")
        if obj.__module__ != module_name:
            print("  - RESULT: Module mismatch. Skipping.")
            continue
        else:
            print("  - RESULT: Module match. OK.")

        # Check 2: Inheritance
        print(f"  - issubclass(obj, BaseTool): {issubclass(obj, BaseTool)}")
        print(f"  - obj is BaseTool: {obj is BaseTool}")
        if not (issubclass(obj, BaseTool) and obj is not BaseTool):
            print("  - RESULT: Inheritance check failed. Skipping.")
            continue
        else:
            print("  - RESULT: Inheritance OK.")

        # Check 3: Signature
        try:
            sig = inspect.signature(obj.__init__)
            print(f"  - __init__ signature: {sig}")
            params = sig.parameters
            missing_deps = False
            for p_name, p in params.items():
                if p_name == 'self': continue
                if p.default == inspect.Parameter.empty:
                    print(f"  - Found required parameter: {p_name}")
                    missing_deps = True
            
            if missing_deps:
                print("  - RESULT: Missing dependencies. Skipping.")
            else:
                print("  - RESULT: Signature OK.")
        except Exception as e:
            print(f"  - Signature check errored: {e}")

        # Check 4: Internal flag
        is_internal = getattr(obj, 'is_internal', False)
        print(f"  - is_internal: {is_internal}")
        if is_internal:
            print("  - RESULT: Marked as internal. Skipping.")
        else:
            print("  - RESULT: Not internal. OK.")

        # Final instantiation test
        try:
            instance = obj()
            print(f"  - Instantiation successful. Tool name: {instance.name}")
        except Exception as e:
            print(f"  - Instantiation FAILED: {e}")

if __name__ == "__main__":
    diagnose()