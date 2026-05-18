import inspect
from aeon.tools.generate_video import GenerateVideoTool

def inspect_tool():
    print("--- Inspecting GenerateVideoTool._get_workflow ---")
    try:
        source = inspect.getsource(GenerateVideoTool._get_workflow)
        print(source)
    except Exception as e:
        print(f"Could not get source: {e}")
    
    # Also test a dummy call to see what it actually returns
    tool = GenerateVideoTool()
    workflow = tool._get_workflow("text_to_video", "test", 768, 512, 33)
    print("\n--- Actual Workflow Output ---")
    import json
    print(json.dumps(workflow, indent=2))

if __name__ == "__main__":
    inspect_tool()