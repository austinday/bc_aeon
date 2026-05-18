import os
import sys
import json
import argparse
import time
from pathlib import Path
from unittest.mock import MagicMock

# Ensure we can import aeon
sys.path.append(os.getcwd())

try:
    from aeon.core.worker import Worker
    from aeon.core.llm import LLMClient
    from aeon.tools.loader import load_tools_from_directory
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("[DEBUG] Starting sub-agent startup trace...")
    
    # Mock configurations
    config = {
        'model': 'test-model',
        'provider': 'local',
        'context_limit': 128000
    }
    
    # Mock LLM Client to avoid actual API calls and return valid JSON
    class MockLLMClient:
        def __init__(self, strong_config, weak_config):
            self.primary_model = strong_config['model']
            self.utility_model = weak_config['model']
        
        def get_primary_agent_response(self, prompt):
            print("[MOCK LLM] Receiving prompt, returning valid JSON action...")
            return json.dumps({
                "thought": "I will complete the task.",
                "actions": [{"tool_name": "task_complete", "parameters": {"reason": "Mock success"}}]
            })
        
        def __init__(self, *args, **kwargs): # Handle potential double init
            pass

    # We need to patch LLMClient before Worker is instantiated
    import aeon.core.llm
    original_llm_client = aeon.core.llm.LLMClient
    
    def mock_llm_factory(strong_config, weak_config):
        return MockLLMClient(strong_config, weak_config)
    
    aeon.core.llm.LLMClient = mock_llm_factory

    try:
        # Simulate the sub_agent_wrapper.py startup sequence
        print("[2026-05-17] Initializing sub-agent test_agent...")
        
        # 1. LLM Client
        llm_client = aeon.core.llm.LLMClient(strong_config=config, weak_config=config)
        
        # 2. Worker
        worker = Worker(
            llm_client=llm_client, 
            debug_mode=True, 
            telemetry_callback=lambda i, t, o: None, 
            steering_path="test_steering.txt"
        )
        
        # 3. Tools loading
        print("[DEBUG] Loading tools...")
        deps = {'llm_client': llm_client}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        
        print("[DEBUG] Registering tools...")
        worker.register_tools(tools)
        
        print("[DEBUG] Starting worker.run...")
        worker.run("Test objective", max_iterations=1)
        
        print("[DEBUG] Worker finished successfully.")

    except Exception as e:
        import traceback
        print("\n--- CRASH TRACEBACK ---")
        traceback.print_exc()
        print("--- END TRACEBACK ---\n")
        sys.exit(1)

if __name__ == "__main__":
    main()