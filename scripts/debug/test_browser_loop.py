import os
import sys
import json
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

def test_end_to_end_browser():
    print("Starting End-to-End Browser Tool-Calling Loop Test...")
    
    # 1. Setup LLM Client
    # Using a strong model for tool calling. 
    # In a real environment, we'd use the menu, but here we hardcode a known available model.
    # I'll try to use a local model if available, otherwise a cloud one.
    # For this test, I'll assume the user has a model configured. 
    # I'll use a generic config that matches the project's LLAMACPP_MODELS or CLOUD_MODELS.
    
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
        # We need to make sure the browser tools are loaded.
        deps = {'llm_client': llm_client, 'worker': worker}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        worker.register_tools(tools)
        
        print(f"Registered {len(worker.tools)} tools.")
        
        # 3. Define a Browser-based Objective
        # This objective requires: Navigation -> Visual Analysis -> Interaction -> Verification
        objective = "Go to https://www.google.com, search for 'Aeon Agent', and tell me the title of the first result. Use the browser tools."
        
        print(f"\nObjective: {objective}")
        print("-" * 50)
        
        # 4. Run the Worker loop
        # We limit the iterations to prevent infinite loops during testing.
        max_iters = 10
        for i in range(max_iters):
            print(f"\n[Iteration {i+1}]")
            # worker.run() typically handles the loop until completion or a stop signal.
            # To observe it step-by-step in a script, we can call the internal logic or just run it.
            # Since worker.run() is the main entry, we'll use it but we might need to 
            # modify it if we want to break out. 
            # For now, let's run it and see if it completes.
            
            # Note: worker.run() internally loops. We'll call it once.
            result = worker.run(objective)
            
            if result:
                print(f"Worker returned result: {result}")
                break
            
            # If worker.run doesn't return but just prints, we'll rely on the logs.
            # In the current implementation, worker.run() runs until the LLM provides a final answer.
            break 

        print("\n" + "-" * 50)
        print("End-to-End Test Completed.")
        return True

    except Exception as e:
        print(f"CRITICAL ERROR during E2E test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Ensure we are in the project root
    os.chdir("/home/aday/bc_aeon")
    success = test_end_to_end_browser()
    if not success:
        sys.exit(1)