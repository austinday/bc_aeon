import aeon.tools.vision
import inspect
import os

def diagnose():
    print("--- VISION TOOL DIAGNOSTIC ---")
    
    # 1. Check file path
    module = aeon.tools.vision
    print(f"Module file: {module.__file__}")
    
    # 2. Inspect the class and method
    try:
        tool_class = module.AnalyzeImageTool
        method = tool_class.execute
        print(f"Method: {method}")
        
        # Get source code of the method
        source = inspect.getsource(method)
        print("\n--- SOURCE CODE OF execute() ---")
        print(source)
        print("--- END SOURCE CODE ---")
        
        # Check for target strings in the source
        targets = [
            "Waiting for vision server to become healthy",
            "Encoding image for analysis",
            "Sending image to Qwen3.6-35B for analysis",
            "Last agent finished vision task",
            "Starting Qwen3.6 vision server"
        ]
        
        print("\n--- STRING SEARCH IN LOADED SOURCE ---")
        for t in targets:
            found = t in source
            print(f"'{t}': {'FOUND' if found else 'NOT FOUND'}")
            
    except Exception as e:
        print(f"Error during inspection: {e}")

if __name__ == "__main__":
    diagnose()