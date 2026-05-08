import os
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.browser import BrowserNavigateTool, BrowserInteractTool, BrowserCloseTabTool
from aeon.core.prompts.manager import PROMPTS_DIR, TOOLS_PROMPTS_DIR, CATS_PROMPTS_DIR

def test_directives():
    # Mock LLM Client
    class MockLLM:
        def __init__(self): self.context_limit = 100000
        def get_primary_agent_response(self, **kwargs): return "{}"
        def set_iteration(self, i): pass
        def analyze_interruption(self, *args): return {}

    # Setup tools
    tools = [
        BrowserNavigateTool(),
        BrowserInteractTool(),
        BrowserCloseTabTool()
    ]
    
    worker = Worker(llm_client=MockLLM(), tools=tools)
    
    print("--- Testing File Creation ---")
    # Check if browser tool files exist
    for tool in tools:
        path = TOOLS_PROMPTS_DIR / f"{tool.name}.txt"
        print(f"File {path.name} exists: {path.exists()}")
        assert path.exists(), f"Prompt file for {tool.name} should have been created"

    print("\n--- Testing Tool Directive Loading ---")
    # Test browser_navigate directives
    nav_directives = worker.tools['browser_navigate'].directives
    print(f"Browser Navigate Directives: {nav_directives}")
    assert len(nav_directives) > 0, "Browser navigate should have loaded directives from file"
    assert "Meticulously verify email parameters" in nav_directives[0]

    print("\n--- Testing Context Injection (Tools) ---")
    # By default, browser tools are in 'web_browser' category, so they aren't active unless expanded
    # But TOP_LEVEL_TOOLS are active. Let's check if they are excluded initially.
    directives_collapsed = worker._get_active_tool_directives()
    print(f"Directives (Collapsed): {directives_collapsed}")
    # Browser tools are NOT top level, so they should NOT be here
    assert "Meticulously verify email parameters" not in directives_collapsed

    print("\n--- Testing Context Injection (Expanded Category) ---")
    worker.expanded_categories.add('web_browser')
    directives_expanded = worker._get_active_tool_directives()
    print(f"Directives (Expanded): {directives_expanded}")
    assert "Meticulously verify email parameters" in directives_expanded
    print("SUCCESS: Directives injected after category expansion")

    print("\n--- Testing Category Prompt Loading ---")
    # Create a dummy category prompt
    cat_path = "web_browser"
    cat_file = CATS_PROMPTS_DIR / f"{cat_path}.txt"
    cat_file.write_text("Category-level directive: Always use a fresh tab for new searches.")
    
    # Re-expand and check
    worker.expanded_categories.remove('web_browser')
    worker.expanded_categories.add('web_browser')
    directives_with_cat = worker._get_active_tool_directives()
    print(f"Directives with Category Prompt: {directives_with_cat}")
    assert "Category-level directive" in directives_with_cat
    print("SUCCESS: Category directives loaded from file")

if __name__ == "__main__":
    try:
        test_directives()
        print("\n\nALL DIRECTIVE TESTS PASSED!")
    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)