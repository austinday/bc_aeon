import os
import sys
import time
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

def test_human_browser_behavior():
    print("Starting Human-Like Browser Behavior Validation...")
    
    # 1. Setup LLM Client (using the known working port from previous iterations)
    strong_config = {
        'model': 'gemma-4-31b-abliterated-Q8_0.gguf',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8013/v1',
        'context_limit': 262144,
    }
    
    try:
        llm_client = LLMClient(strong_config=strong_config, weak_config=strong_config)
        worker = Worker(llm_client=llm_client, debug_mode=True)
        
        # 2. Load and Register Tools
        deps = {'llm_client': llm_client, 'worker': worker}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        worker.register_tools(tools)
        
        print(f"Registered {len(worker.tools)} tools.")
        
        # 3. Test Case 1: Human-like Typing and Tab Preservation
        # We want the agent to type something, switch tabs, and then come back to verify it's still there.
        objective_1 = (
            "1. Navigate to https://www.google.com in tab 'human_test'. "
            "2. Find the search box and type 'Aeon Human Interaction Test' using the browser_interact tool. "
            "3. Do NOT press enter. "
            "4. Navigate to https://www.wikipedia.org in a new tab 'wiki_test'. "
            "5. Switch back to tab 'human_test' and verify that the text 'Aeon Human Interaction Test' is still in the search box."
        )
        
        print(f"\n[Test 1] Objective: {objective_1}")
        print("-" * 50)
        
        # We run the worker. Since we want to verify the *behavior*, we'll check the logs/output.
        # We use a timeout or iteration limit.
        result_1 = worker.run(objective_1)
        print(f"Result 1: {result_1}")

        # 4. Test Case 2: Hover and Visual Inspection
        objective_2 = (
            "1. Navigate to https://www.wikipedia.org in tab 'wiki_test'. "
            "2. Find a link in the sidebar or main page and use the 'hover' action on it. "
            "3. Describe what happens or what is visible after the hover."
        )
        
        print(f"\n[Test 2] Objective: {objective_2}")
        print("-" * 50)
        result_2 = worker.run(objective_2)
        print(f"Result 2: {result_2}")

        print("\n" + "="*50)
        print("Human-Like Behavior Tests Completed.")
        print("="*50)
        return True

    except Exception as e:
        print(f"CRITICAL ERROR during human behavior test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Ensure we are in the project root
    os.chdir("/home/aday/bc_aeon")
    success = test_human_browser_behavior()
    if not success:
        sys.exit(1)