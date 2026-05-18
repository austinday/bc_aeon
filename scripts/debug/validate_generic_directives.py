import os
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.core.prompts.manager import TOOLS_PROMPTS_DIR, CATS_PROMPTS_DIR

def test_generic_directives():
    # 1. Setup: Create dummy prompt files
    # We use a mock LLM client because Worker needs one
    class MockLLM:
        def __init__(self): self.context_limit = 100000
        def set_iteration(self, i): pass
        def get_primary_agent_response(self, **kwargs): return "{}"
        def analyze_interruption(self, **kwargs): return {}

    # Define test data
    test_tools = {
        "tool_a": "Directive A1\nDirective A2",
        "tool_b": "", # Empty file
        "tool_c": "Directive C1"
    }
    test_cats = {
        "cat_x": "Category X Directive",
        "cat_y": "" # Empty file
    }

    # Ensure directories exist
    TOOLS_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    CATS_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Write tool prompts
        for name, content in test_tools.items():
            (TOOLS_PROMPTS_DIR / f"{name}.txt").write_text(content)
        
        # Write category prompts
        for name, content in test_cats.items():
            (CATS_PROMPTS_DIR / f"{name}.txt").write_text(content)

        # 2. Initialize Worker
        # We need to provide a list of tools. We'll create dummy tool objects.
        class DummyTool:
            def __init__(self, name):
                self.name = name
                self.description = "desc"
        
        tools_list = [DummyTool(name) for name in test_tools.keys()]
        worker = Worker(llm_client=MockLLM(), tools=tools_list)
        
        # Expand categories for testing
        worker.expanded_categories = set(test_cats.keys())

        # 3. Execute and Validate
        directives = worker._get_active_tool_directives()
        print(f"Generated Directives:\n{directives}\n")

        # Check Tool A (Multiple lines)
        assert "- tool_a: Directive A1" in directives
        assert "- tool_a: Directive A2" in directives
        
        # Check Tool B (Empty)
        assert "- tool_b: " in directives
        
        # Check Tool C (Single line)
        assert "- tool_c: Directive C1" in directives
        
        # Check Cat X (Content)
        assert "- cat_x: Category X Directive" in directives
        
        # Check Cat Y (Empty)
        assert "- cat_y: " in directives

        print("✅ Validation Successful: Directives are loaded generically from files.")
        return True

    except Exception as e:
        print(f"❌ Validation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        for name in test_tools.keys():
            (TOOLS_PROMPTS_DIR / f"{name}.txt").unlink(missing_ok=True)
        for name in test_cats.keys():
            (CATS_PROMPTS_DIR / f"{name}.txt").unlink(missing_ok=True)

if __name__ == "__main__":
    if test_generic_directives():
        exit(0)
    else:
        exit(1)