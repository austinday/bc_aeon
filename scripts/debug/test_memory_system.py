import sys
import os

# Add the root directory to sys.path to allow importing aeon
sys.path.append(os.getcwd())

from aeon.core.worker import Worker
from aeon.tools.memory import MemorizeTool, ForgetTool, ListMemoriesTool

class MockLLMClient:
    def __init__(self):
        self.context_limit = 128000
    def set_iteration(self, i): pass
    def get_primary_agent_response(self, prompt, diagnostic_str): return "{}"
    def analyze_interruption(self, obj, text): return {"classification": "ADVICE"}
    def compress_action_log(self, text): return text
    def compress_memories(self, text): return {}

def test_memory_system():
    print("Testing Enhanced Memory System...")
    
    # Setup
    llm = MockLLMClient()
    # We need to pass tools to Worker, but we'll manually add the memory tools for the test
    worker = Worker(llm_client=llm, tools=[])
    
    mem_tool = MemorizeTool(worker)
    forget_tool = ForgetTool(worker)
    list_tool = ListMemoriesTool(worker)
    
    worker.tools["memorize"] = mem_tool
    worker.tools["forget"] = forget_tool
    worker.tools["list_memories"] = list_tool

    # 1. Test Memorize
    print("\n--- Testing Memorize ---")
    res1 = mem_tool.execute(key="project_goal", value="Build a rocket", category="planning")
    res2 = mem_tool.execute(key="api_key", value="12345", category="credentials")
    res3 = mem_tool.execute(key="temp_note", value="Check logs", category="general")
    
    assert "project_goal" in worker.memories
    assert worker.memories["project_goal"]["category"] == "planning"
    assert worker.memories["project_goal"]["value"] == "Build a rocket"
    print("Memorize: SUCCESS")

    # 2. Test ListMemories
    print("\n--- Testing ListMemories ---")
    all_mems = list_tool.execute()
    assert "project_goal" in all_mems
    assert "api_key" in all_mems
    
    cat_mems = list_tool.execute(category="planning")
    assert "project_goal" in cat_mems
    assert "api_key" not in cat_mems
    print("ListMemories: SUCCESS")

    # 3. Test Forget (Key)
    print("\n--- Testing Forget (Key) ---")
    forget_tool.execute(key="temp_note")
    assert "temp_note" not in worker.memories
    print("Forget Key: SUCCESS")

    # 4. Test Forget (Category)
    print("\n--- Testing Forget (Category) ---")
    forget_tool.execute(category="credentials")
    assert "api_key" not in worker.memories
    assert "project_goal" in worker.memories # Should still be there
    print("Forget Category: SUCCESS")

    # 5. Test Worker._format_memories
    print("\n--- Testing Worker._format_memories ---")
    formatted = worker._format_memories()
    assert "[planning] project_goal: Build a rocket" in formatted
    print("Format Memories: SUCCESS")

    print("\nALL MEMORY TESTS PASSED!")

if __name__ == "__main__":
    try:
        test_memory_system()
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)