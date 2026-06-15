import sys
import os

# Ensure we can import from the aeon package
sys.path.insert(0, os.getcwd())

try:
    from aeon.core.worker import Worker
    from aeon.core.llm import LLMClient
    from aeon.tools.loader import load_tools_from_directory
    from aeon.tools.skills_manager_tool import ExpandSkillsCategory, CollapseSkillsCategory
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_registry():
    print("--- Registry Diagnostic ---")
    
    # Mock configs to satisfy LLMClient
    strong_config = {'provider': 'local', 'model': 'test-model'}
    weak_config = {'provider': 'local', 'model': 'test-model'}
    
    try:
        llm_client = LLMClient(strong_config=strong_config, weak_config=weak_config)
        worker = Worker(llm_client=llm_client)
        
        # Simulate the manual registration in main.py
        tools = []
        try:
            manual_tools = [
                ExpandSkillsCategory(worker=worker, llm_client=llm_client),
                CollapseSkillsCategory(worker=worker, llm_client=llm_client)
            ]
            tools.extend(manual_tools)
            print("[DEBUG] Manually adding skill manager tools to list...")
        except Exception as e:
            print(f"[ERROR] Failed to instantiate manual tools: {e}")

        worker.register_tools(tools)
        
        registered_tools = list(worker.tools.keys())
        print(f"Registered Tools: {registered_tools}")
        
        if 'expand_skills_category' in registered_tools:
            print("\nSUCCESS: 'expand_skills_category' is registered in the Worker.")
        else:
            print("\nFAILURE: 'expand_skills_category' is NOT registered in the Worker.")
            
    except Exception as e:
        print(f"Diagnostic script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_registry()