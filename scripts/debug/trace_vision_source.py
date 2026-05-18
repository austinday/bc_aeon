import inspect
import os
from aeon.tools.vision import AnalyzeImageTool

def main():
    tool = AnalyzeImageTool()
    method = tool.execute
    
    print("--- VISION TOOL SOURCE TRACE ---")
    try:
        source_file = inspect.getfile(method)
        print(f"Source File: {source_file}")
        print(f"Absolute Path: {os.path.abspath(source_file)}")
    except Exception as e:
        print(f"Could not get file: {e}")

    try:
        source_code = inspect.getsource(method)
        print("\n--- SOURCE CODE OF execute() ---")
        print(source_code)
        print("--- END SOURCE CODE ---")
    except Exception as e:
        print(f"Could not get source: {e}")

if __name__ == "__main__":
    main()