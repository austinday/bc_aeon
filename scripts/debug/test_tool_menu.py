import sys
import os

# Add the root directory to sys.path to allow imports from aeon
sys.path.append(os.getcwd())

try:
    from aeon.core.worker import Worker
    from aeon.core.llm import LLMClient
    from aeon.tools.categories import TOOL_CATEGORIES
    
    # Mock LLMClient to avoid network calls
    class MockLLM:
        def __init__(self, *args, **kwargs): pass
        def set_iteration(self, i): pass
        def get_primary_agent_response(self, **kwargs): return "{}"

    # Create a dummy tool to satisfy Worker's tool registry
    class DummyTool:
        def __init__(self, name, description):
            self.name = name
            self.description = description

    # Setup tools
    tools_list = [
        DummyTool('generate_video', 'Generates a video using AI'),
        DummyTool('generate_image', 'Generates an image using AI'),
        DummyTool('browser_navigate', 'Navigates to a URL'),
    ]

    # Instantiate Worker
    worker = Worker(llm_client=MockLLM())
    worker.register_tools(tools_list)

    print("--- Testing _get_tools_description ---")
    description = worker._get_tools_description()
    print(description)
    
    if "video_tools" in description:
        print("\nRESULT: 'video_tools' FOUND in description.")
    else:
        print("\nRESULT: 'video_tools' NOT FOUND in description.")

    print("\n--- Testing _render_categories ---")
    # Manually call render_categories
    lines = worker._render_categories(TOOL_CATEGORIES, '', 0)
    for line in lines:
        print(line)

except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()