import sys
import os
from pathlib import Path

# Ensure the project root is in the path
sys.path.insert(0, os.getcwd())

try:
    from aeon.tools.loader import load_tools_from_directory
    from aeon.tools.base import BaseTool
    import aeon.tools.skills_manager_tool as smt
    
    print("--- Tool Loading Debug ---")
    
    # Test 1: Check if the classes are actually BaseTool subclasses
    for name in ["ExpandSkillsCategory", "CollapseSkillsCategory"]:
        cls = getattr(smt, name)
        print(f"Checking {name}:")
        print(f"  - Is class: {isinstance(cls, type)}")
        print(f"  - Is subclass of BaseTool: {issubclass(cls, BaseTool)}")
        print(f"  - Module: {cls.__module__}")

    # Test 2: Try loading via the actual loader
    print("\nRunning load_tools_from_directory...")
    # We provide an empty dict for dependencies since these tools don't seem to need any in __init__
    tools = load_tools_from_directory("aeon.tools", dependencies={}, verbose=True)
    
    tool_names = [t.name for t in tools]
    print(f"\nLoaded tool names: {tool_names}")
    
    if "expand_skills_category" in tool_names and "collapse_skills_category" in tool_names:
        print("\nRESULT: SUCCESS - Tools were loaded correctly.")
    else:
        print("\nRESULT: FAILURE - Tools were not loaded.")
        
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nERROR: {e}")