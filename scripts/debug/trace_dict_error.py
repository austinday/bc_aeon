import os
import sys
import json
import traceback
from unittest.mock import MagicMock

# Force PYTHONPATH to include current dir
sys.path.append(os.getcwd())

try:
    from aeon.core.worker import Worker
    from aeon.core.llm import LLMClient
    from aeon.tools.loader import load_tools_from_directory
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def debug_startup():
    print("[DEBUG] Starting surgical startup trace...")
    
    # Mock LLM config
    config = {'model': 'test-model', 'provider': 'local', 'context_limit': 1000}
    
    try:
        # 1. Setup LLM Client
        llm_client = LLMClient(strong_config=config, weak_config=config)
        
        # 2. Setup Worker
        worker = Worker(llm_client=llm_client)
        
        # 3. Load Tools
        print("[DEBUG] Loading tools...")
        deps = {'llm_client': llm_client}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps, verbose=False)
        print(f"[DEBUG] Loaded {len(tools)} tools.")
        
        # 4. Register Tools - This is where we suspect the crash
        print("[DEBUG] Registering tools...")
        worker.register_tools(tools)
        print("[DEBUG] Registration successful.")
        
        print("[DEBUG] Startup sequence completed without crash.")
        
    except TypeError as e:
        print(f"\n{ '!'*20 }\nCAUGHT TYPEERROR:\n{e}\n{ '!'*20 }")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    debug_startup()