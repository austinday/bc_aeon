import aeon
import sys
import os
from pathlib import Path

def check_env():
    print(f"--- RUNTIME ENVIRONMENT CHECK ---")
    print(f"Python Executable: {sys.executable}")
    print(f"Aeon Package Path: {aeon.__file__}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Sys Path: {sys.path}")
    
    # Check if we are in the expected project root
    expected_root = "/home/aday/bc_aeon/aeon"
    actual_path = aeon.__file__
    if actual_path.startswith(expected_root):
        print(f"SUCCESS: Aeon is loaded from the local source directory.")
    else:
        print(f"WARNING: Aeon is loaded from {actual_path}, NOT the local source!")

    # Try to find the 'REJECTED' string in the actual loaded module object
    from aeon.tools import loader
    import inspect
    source = inspect.getsource(loader.load_tools_from_directory)
    if "LOADER DEBUG" in source or "REJECTED" in source:
        print("CRITICAL: The LOADED module still contains debug strings!")
        print("--- LOADED SOURCE ---")
        print(source)
    else:
        print("The LOADED module source is clean.")

if __name__ == "__main__":
    check_env()