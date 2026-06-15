import sys
import os

# Mocking dependencies to allow the script to run without a full LLM setup
class MockLLM:
    def __init__(self, **kwargs): pass
    def get_primary_agent_response(self, **kwargs): return "{}"

class MockWorker:
    def __init__(self, **kwargs):
        self.tools = {}
    def register_tools(self, tools):
        for t in tools:
            self.tools[t.name] = t

# Import the actual components
sys.path.append(os.getcwd())
try:
    from aeon.core.worker import Worker
    from aeon.tools.skills_manager_tool import ExpandSkillsCategory, CollapseSkillsCategory
    
    print("--- STARTING REGISTRY VERIFICATION ---")
    
    # Simulate the manual registration in main.py
    worker = Worker(llm_client=MockLLM())
    
    # Manually add the tools as done in main.py
    tools = [
        ExpandSkillsCategory(worker=worker),
        CollapseSkillsCategory(worker=worker)
    ]
    worker.register_tools(tools)
    
    print(f"Registered tools: {list(worker.tools.keys())}")
    
    target = "expand_skills_category"
    if target in worker.tools:
        print(f"SUCCESS: '{target}' is present in the registry.")
    else:
        print(f"FAILURE: '{target}' NOT found. Available keys: {list(worker.tools.keys())}")

except Exception as e:
    print(f"Error during verification: {e}")
    import traceback
    traceback.print_exc()