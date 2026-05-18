import sys
import os
from unittest.mock import MagicMock, patch

# Add current directory to path to import aeon
sys.path.append(os.getcwd())

import openai
from aeon.core.llm import LLMClient

def test_recovery_logic():
    print("Testing LLMClient recovery logic for InternalServerError and APIConnectionError...")

    # 1. Setup Configs
    strong_config = {'provider': 'local', 'model': 'strong-model', 'context_limit': 128000}
    weak_config = {'provider': 'local', 'model': 'weak-model', 'context_limit': 128000}
    
    client = LLMClient(strong_config, weak_config)
    
    # 2. Create Mock Exceptions using spec to pass isinstance checks without constructor issues
    # InternalServerError is a subclass of APIStatusError
    mock_internal_error = MagicMock(spec=openai.InternalServerError)
    mock_connection_error = MagicMock(spec=openai.APIConnectionError)

    # 3. Mock the primary client's chat completions
    # We want it to fail twice (one of each error) and then succeed
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"thought": "Recovered!", "actions": [{"tool_name": "task_complete", "parameters": {"reason": "Success"}}]}'
    
    # Mock the stream for get_primary_agent_response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = '{"thought": "Recovered!", "actions": [{"tool_name": "task_complete", "parameters": {"reason": "Success"}}]}'
    
    # The actual call in get_primary_agent_response is: self.primary_client.chat.completions.create(...)
    # It returns a stream (generator)
    client.primary_client.chat.completions.create = MagicMock(
        side_effect=[
            mock_internal_error,   # First call: InternalServerError
            mock_connection_error, # Second call: APIConnectionError
            [mock_chunk]          # Third call: Success (stream)
        ]
    )

    # 4. Mock the recovery check: primary_client.models.list()
    # It should fail a few times then succeed to test the loop in _handle_connection_error
    client.primary_client.models = MagicMock()
    client.primary_client.models.list = MagicMock(
        side_effect=[
            Exception("Server still loading..."),
            Exception("Still not ready..."),
            MagicMock() # Finally succeeds
        ]
    )

    print("\n--- Executing get_primary_agent_response ---")
    try:
        # We use a simple prompt. The client should:
        # 1. Call create -> get InternalServerError -> call _handle_connection_error
        # 2. _handle_connection_error calls models.list() -> fails, fails, succeeds -> returns True
        # 3. Retry create -> get APIConnectionError -> call _handle_connection_error
        # 4. _handle_connection_error calls models.list() -> succeeds immediately -> returns True
        # 5. Retry create -> get Success -> return JSON
        
        result = client.get_primary_agent_response("Hello")
        print(f"\nFinal Result: {result}")
        
        if "Recovered!" in result:
            print("\nTEST SUITE PASSED: LLMClient successfully recovered from both InternalServerError and APIConnectionError.")
        else:
            print("\nTEST SUITE FAILED: Result did not contain expected recovery text.")
            sys.exit(1)

    except Exception as e:
        print(f"\nTEST SUITE FAILED: Unexpected exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_recovery_logic()