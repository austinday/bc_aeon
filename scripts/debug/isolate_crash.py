import sys
import json
import traceback
from pathlib import Path

# Ensure we can import aeon
sys.path.append(".")

try:
    from aeon.core.worker import Worker
    from aeon.core.llm import LLMClient
    from aeon.tools.loader import load_tools_from_directory

    print("[DEBUG] Imports successful.")

    # Mock config similar to what sub_agent_wrapper would pass
    config = {
        'model': 'test-model',
        'provider': 'local',
        'context_limit': 128000,
        'base_url': 'http://localhost:8013/v1'
    }

    print("[DEBUG] Testing LLMClient instantiation...")
    # We mock the OpenAI client to avoid actual network calls
    import openai
    openai.OpenAI = lambda **kwargs: type('MockClient', (), {'models': type('MockModels', (), {'list': lambda: []})()})()

    llm = LLMClient(strong_config=config, weak_config=config)
    print("[DEBUG] LLMClient instantiated successfully.")

    print("[DEBUG] Testing Worker instantiation...")
    worker = Worker(llm_client=llm)
    print("[DEBUG] Worker instantiated successfully.")

    print("[DEBUG] Testing tool loading...")
    # Mock dependencies
    deps = {'llm_client': llm}
    tools = load_tools_from_directory("aeon.tools", dependencies=deps, verbose=False)
    print(f"[DEBUG] Loaded {len(tools)} tools.")

    print("[DEBUG] Testing worker.register_tools(tools)...")
    worker.register_tools(tools)
    print("[DEBUG] register_tools completed successfully.")

    print("\n[RESULT] No crash detected in basic startup sequence.")

except Exception as e:
    print(f"\n[CRASH] Caught exception: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)