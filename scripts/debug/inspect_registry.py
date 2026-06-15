import sys
import os

# Mocking dependencies to allow the script to run without a full agent boot
class MockLLM:
    def __init__(self, *args, **kwargs): pass
    def get_primary_agent_response(self, **kwargs): return "{}"

class MockWorker:
    def __init__(self, *args, **kwargs):
        self.tools = {}
    def register_tools(self, tools):
        for t in tools:
            self.tools[t.name] = t

# Import the actual tools
try:
    sys.path.append(os.getcwd())
    from aeon.tools.skills_manager_tool import ExpandSkillsCategory, CollapseSkillsCategory
    
    print("--- Tool Registration Test ---")
    # Simulate the main.py registration process
    worker = MockWorker()
    manual_tools = [
        ExpandSkillsCategory(worker=None, llm_client=None),
        CollapseSkillsCategory(worker=None, llm_client=None)
    ]
    worker.register_tools(manual_tools)
    
    print(f"Registered tools: {list(worker.tools.keys())}")
    
    if 'expand_skills_category' in worker.tools:
        print("SUCCESS: 'expand_skills_category' is registered.")
    else:
        print("FAILURE: 'expand_skills_category' NOT found in registry.")
        for name, tool in worker.tools.items():
            print(f"Found tool: {name} (Class: {tool.__class__.__name__})")

except Exception as e:
    print(f"Error during diagnostic: {e}")
    import traceback
    traceback.print_exc()