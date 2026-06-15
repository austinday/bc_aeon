import sys
import os

# Ensure we can import from the aeon package
sys.path.insert(0, os.getcwd())

try:
    from aeon.core.worker import Worker
    from aeon.core.llm import LLMClient
    from aeon.tools.skills_manager_tool import ExpandSkillsCategory, CollapseSkillsCategory
    
    print("--- Registry Diagnostic ---")
    
    # Mock LLMClient to avoid needing real config/API keys
    class MockLLMClient:
        def __init__(self, *args, **kwargs):
            self.context_limit = 128000
        def get_primary_agent_response(self, **kwargs):
            return "{}"

    llm_client = MockLLMClient()
    worker = Worker(llm_client=llm_client)
    
    # Simulate the manual registration in main.py
    manual_tools = [
        ExpandSkillsCategory(worker=worker, llm_client=llm_client),
        CollapseSkillsCategory(worker=worker, llm_client=llm_client)
    ]
    
    worker.register_tools(manual_tools)
    
    print(f"Worker instance ID: {worker.instance_id}")
    print(f"Registered tools: {list(worker.tools.keys())}")
    
    if 'expand_skills_category' in worker.tools:
        print("SUCCESS: 'expand_skills_category' is present in the registry.")
        tool = worker.tools['expand_skills_category']
        print(f"Tool class: {type(tool)}")
        print(f"Tool name attribute: {tool.name}")
    else:
        print("FAILURE: 'expand_skills_category' is NOT in the registry.")

except Exception as e:
    import traceback
    traceback.print_exc()