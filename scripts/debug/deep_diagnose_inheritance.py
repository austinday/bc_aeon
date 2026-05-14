import sys
import importlib
from aeon.tools.base import BaseTool as BaseToolExpected

print("=== Deep Inheritance Diagnosis ===")
print(f"Expected BaseTool: {BaseToolExpected}")
print(f"Expected BaseTool ID: {id(BaseToolExpected)}")
print(f"Expected BaseTool Module: {BaseToolExpected.__module__}")

try:
    import aeon.tools.generate_video as gv_mod
    GenerateVideoTool = gv_mod.GenerateVideoTool
    print(f"\nTarget Class: {GenerateVideoTool}")
    print(f"Target Class Module: {GenerateVideoTool.__module__}")
    
    print("\n--- MRO Analysis ---")
    for cls in GenerateVideoTool.__mro__:
        print(f"Class: {cls}")
        print(f"  ID: {id(cls)}")
        print(f"  Module: {cls.__module__}")
        if cls.__name__ == 'BaseTool':
            print(f"  MATCH FOUND: This is a 'BaseTool' class. Is it the expected one? {cls is BaseToolExpected}")

    print("\n--- Subclass Check ---")
    print(f"issubclass(GenerateVideoTool, BaseToolExpected): {issubclass(GenerateVideoTool, BaseToolExpected)}")

except Exception as e:
    print(f"Error during analysis: {e}")

print("\n--- sys.modules Investigation ---")
base_tool_modules = [m for m in sys.modules if 'tools.base' in m]
print(f"Modules matching 'tools.base': {base_tool_modules}")

for m_name in base_tool_modules:
    try:
        mod = sys.modules[m_name]
        if hasattr(mod, 'BaseTool'):
            print(f"Module {m_name} has BaseTool with ID: {id(mod.BaseTool)}")
    except Exception:
        pass

if len(base_tool_modules) > 1:
    print("\nCRITICAL: Duplicate BaseTool modules detected in sys.modules!")
else:
    print("\nNo obvious duplicate modules found in sys.modules.")