import importlib
import sys
import os
from aeon.tools.generate_video import GenerateVideoTool
from aeon.tools.base import BaseTool

def trace():
    print("=== Module Path Trace ===")
    
    # Get the module where GenerateVideoTool is defined
    module = sys.modules.get('aeon.tools.generate_video')
    if module:
        print(f"Module 'aeon.tools.generate_video' is loaded from: {getattr(module, '__file__', 'Unknown')}")
    else:
        print("Module 'aeon.tools.generate_video' not found in sys.modules")

    # Get the module where BaseTool is defined
    base_module = sys.modules.get('aeon.tools.base')
    if base_module:
        print(f"Module 'aeon.tools.base' is loaded from: {getattr(base_module, '__file__', 'Unknown')}")
    else:
        print("Module 'aeon.tools.base' not found in sys.modules")

    print("\n=== Inheritance Check ===")
    print(f"GenerateVideoTool MRO: {GenerateVideoTool.__mro__}")
    print(f"Is subclass of BaseTool: {issubclass(GenerateVideoTool, BaseTool)}")
    
    print("\n=== Environment Check ===")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
    print(f"sys.path: {sys.path}")

if __name__ == "__main__":
    trace()