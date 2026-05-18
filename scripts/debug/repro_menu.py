from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

# Mock LLMClient to avoid needing actual API keys/configs
class MockLLMClient(LLMClient):
    def __init__(self, *args, **kwargs):
        self.context_limit = 128000
        self.set_iteration = lambda x: None
    def get_primary_agent_response(self, **kwargs):
        return "{}"

def test_menu():
    print("Initializing Worker...")
    llm = MockLLMClient()
    worker = Worker(llm_client=llm)
    
    print("Loading tools...")
    # Pass empty dependencies as we just want to see the menu
    tools = load_tools_from_directory("aeon.tools", dependencies={})
    worker.register_tools(tools)
    
    print("\n--- STARTUP MENU OUTPUT ---")
    menu = worker._get_tools_description()
    print(menu)
    print("--- END MENU OUTPUT ---\n")
    
    if "video_tools" in menu:
        print("SUCCESS: 'video_tools' found in the menu.")
    else:
        print("FAILURE: 'video_tools' NOT found in the menu.")

if __name__ == "__main__":
    test_menu()