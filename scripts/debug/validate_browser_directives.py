import sys
from unittest.mock import MagicMock
from aeon.core.worker import Worker

class MockTool:
    def __init__(self, name, directives=None):
        self.name = name
        self.description = f"Description for {name}"
        self.directives = directives or []

def test_browser_filtering():
    # Mock LLMClient
    mock_llm = MagicMock()
    
    # Create a set of tools: some browser, some not
    tools = [
        MockTool("browser_navigate", ["Navigate directive"]),
        MockTool("browser_interact", ["Interact directive"]),
        MockTool("file_read", ["File read directive"]),
        MockTool("system_info", ["System info directive"]),
        MockTool("other_tool", ["Other directive"]),
    ]
    
    # Initialize Worker
    worker = Worker(llm_client=mock_llm, tools=tools)
    
    # Mock expanded categories
    worker.expanded_categories = {"web_browser", "image_tools"}
    
    # Mock load_cat_prompt because it reads from disk
    import aeon.core.prompts.manager as prompt_manager
    original_load_cat = prompt_manager.load_cat_prompt
    
    def mocked_load_cat(path):
        if path == "web_browser":
            return ["Web Browser Category Directive"]
        if path == "image_tools":
            return ["Image Tools Category Directive"]
        return []
    
    prompt_manager.load_cat_prompt = mocked_load_cat
    
    try:
        directives_str = worker._get_active_tool_directives()
        print(f"Resulting Directives:\n{directives_str}")
        
        # 1. Verify browser_navigate IS populated
        assert "browser_navigate: Navigate directive" in directives_str, "browser_navigate should be populated"
        
        # 2. Verify other tools are PRESENT but NOT populated
        # The format is "- tool_name: "
        assert "- browser_interact: " in directives_str, "browser_interact should be listed"
        assert "Interact directive" not in directives_str, "browser_interact should NOT be populated"
        
        assert "- file_read: " in directives_str, "file_read should be listed"
        assert "File read directive" not in directives_str, "file_read should NOT be populated"
        
        assert "- other_tool: " in directives_str, "other_tool should be listed"
        assert "Other directive" not in directives_str, "other_tool should NOT be populated"
        
        # 3. Verify categories are PRESENT but NOT populated
        assert "- web_browser: " in directives_str, "web_browser category should be listed"
        assert "Web Browser Category Directive" not in directives_str, "web_browser category should NOT be populated"
        
        assert "- image_tools: " in directives_str, "image_tools category should be listed"
        assert "Image Tools Category Directive" not in directives_str, "image_tools category should NOT be populated"
        
        print("\n✅ VALIDATION SUCCESSFUL: Only browser_navigate is populated; all others are empty.")
    finally:
        prompt_manager.load_cat_prompt = original_load_cat

if __name__ == "__main__":
    try:
        test_browser_filtering()
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)