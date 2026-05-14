import sys
import os

# Force the current working directory to the front of sys.path
# This ensures we import the local version of the code, not an installed package version.
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

print(f"Using sys.path: {sys.path[0]}")

try:
    from aeon.tools.generate_video import GenerateVideoTool
    from aeon.tools.base import BaseTool
    print("Successfully imported GenerateVideoTool and BaseTool.")
    print(f"GenerateVideoTool module: {GenerateVideoTool.__module__}")
    print(f"BaseTool module: {BaseTool.__module__}")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_video_tool():
    print("\nTesting GenerateVideoTool instantiation...")
    try:
        tool = GenerateVideoTool()
        print(f"Tool instantiated successfully: {tool}")
        
        print(f"GenerateVideoTool MRO: {GenerateVideoTool.__mro__}")
        
        # Check inheritance
        if isinstance(tool, BaseTool):
            print("Verification SUCCESS: GenerateVideoTool inherits from BaseTool.")
        else:
            print("Verification FAILURE: GenerateVideoTool does NOT inherit from BaseTool.")
            # Debugging the identity
            print(f"Tool class: {type(tool)}")
            print(f"BaseTool class: {BaseTool}")
            sys.exit(1)
            
        # Check required attributes
        if hasattr(tool, 'name') and tool.name == "generate_video":
            print("Verification SUCCESS: Tool name is correct.")
        else:
            print(f"Verification FAILURE: Tool name is incorrect or missing. Found: {getattr(tool, 'name', 'None')}")
            sys.exit(1)
            
        if hasattr(tool, 'description') and tool.description:
            print("Verification SUCCESS: Tool description is present.")
        else:
            print("Verification FAILURE: Tool description is missing.")
            sys.exit(1)
            
        print("\nAll standalone checks passed!")
    except Exception as e:
        print(f"Unexpected error during tool instantiation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_video_tool()