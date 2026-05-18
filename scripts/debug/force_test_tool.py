import sys
import os
import importlib
from pathlib import Path

# Force the local directory into the path
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

print(f"[DIAGNOSTIC] Python Path: {sys.path}")

try:
    import aeon.tools.generate_video as gv_module
    importlib.reload(gv_module)
    print(f"[DIAGNOSTIC] Successfully imported and reloaded {gv_module.__file__}")
    
    from aeon.tools.generate_video import GenerateVideoTool
    print(f"[DIAGNOSTIC] Class GenerateVideoTool found. Methods: {dir(GenerateVideoTool)}")
    
    has_execute = 'execute' in dir(GenerateVideoTool)
    print(f"[DIAGNOSTIC] Has 'execute' method: {has_execute}")
    
    if has_execute:
        print("[DIAGNOSTIC] Attempting instantiation...")
        try:
            tool = GenerateVideoTool()
            print("[DIAGNOSTIC] Instantiation SUCCESS")
        except TypeError as e:
            print(f"[DIAGNOSTIC] Instantiation FAILED: {e}")
    else:
        print("[DIAGNOSTIC] 'execute' method is missing from the class definition!")

except Exception as e:
    print(f"[DIAGNOSTIC] CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()