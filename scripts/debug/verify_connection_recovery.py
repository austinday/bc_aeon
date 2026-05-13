import os
import time
import json
import openai
from unittest.mock import MagicMock, patch
from aeon.core.llm import LLMClient

# Mock configurations
strong_config = {
    'provider': 'local',
    'model': 'strong-model',
    'context_limit': 128000
}
weak_config = {
    'provider': 'local',
    'model': 'weak-model',
    'context_limit': 128000
}

def test_connection_recovery():
    print("Starting Connection Recovery Verification Test...")
    
    # Initialize LLMClient
    client = LLMClient(strong_config, weak_config)
    
    # We want to simulate:
    # 1. A call to get_primary_agent_response fails with APIConnectionError.
    # 2. The recovery loop starts.
    # 3. The first few recovery checks (models.list()) also fail.
    # 4. Eventually, the recovery check succeeds.
    # 5. The original request is retried and succeeds.

    # Mock the primary client
    mock_openai_client = MagicMock()
    client.primary_client = mock_openai_client
    
    # Setup the failure sequence for the main request
    # First call: Raise ConnectionError
    # Second call: Return a valid response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].delta.content = '{"thought": "Recovered!", "actions": [{"tool_name": "task_complete", "parameters": {"reason": "Verified"}}]}'
    
    # We use a side_effect to simulate the sequence of events
    # Note: get_primary_agent_response uses streaming for non-Vertex clients
    # So we need to mock the stream.
    
    class MockStream:
        def __init__(self, content, fail_count):
            self.content = content
            self.fail_count = fail_count
            self.called = 0

        def __iter__(self):
            if self.fail_count > 0:
                self.fail_count -= 1
                raise openai.APIConnectionError("Connection failed")
            
            # Simulate streaming chunks
            for char in self.content:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = char
                yield chunk

    # Mock for the recovery check (models.list())
    # Fail 3 times, then succeed
    recovery_fail_count = 3
    def recovery_check_side_effect():
        nonlocal recovery_fail_count
        if recovery_fail_count > 0:
            recovery_fail_count -= 1
            print(f"  [Mock] Recovery check failed (Remaining: {recovery_fail_count})")
            raise openai.APIConnectionError("Still no connection")
        print("  [Mock] Recovery check succeeded!")
        return MagicMock()

    mock_openai_client.models.list.side_effect = recovery_check_side_effect

    # Mock the chat.completions.create for the main request
    # First call fails, second call returns the stream
    main_request_fail_count = 1
    def main_request_side_effect(*args, **kwargs):
        nonlocal main_request_fail_count
        if main_request_fail_count > 0:
            main_request_fail_count -= 1
            print("  [Mock] Main request failed with APIConnectionError")
            raise openai.APIConnectionError("Connection failed")
        
        print("  [Mock] Main request succeeded, returning stream")
        return MockStream('{"thought": "Recovered!", "actions": [{"tool_name": "task_complete", "parameters": {"reason": "Verified"}}]}', 0)

    mock_openai_client.chat.completions.create.side_effect = main_request_side_effect

    print("\nExecuting get_primary_agent_response...")
    start_time = time.time()
    try:
        result = client.get_primary_agent_response("Test prompt")
        end_time = time.time()
        
        print(f"\nResult: {result}")
        print(f"Total time elapsed: {end_time - start_time:.2f}s")
        
        # Verify the result is the expected JSON
        parsed = json.loads(result)
        if parsed.get("thought") == "Recovered!":
            print("\nVERIFICATION SUCCESS: Agent recovered from connection error and resumed.")
        else:
            print("\nVERIFICATION FAILURE: Unexpected result.")
            exit(1)
            
    except Exception as e:
        print(f"\nVERIFICATION FAILURE: Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    test_connection_recovery()