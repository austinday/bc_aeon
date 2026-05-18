import sys
import os
from unittest.mock import MagicMock

# Ensure the local aeon package is importable
sys.path.append(os.getcwd())

try:
    from aeon.tools.search import SummarizeTopicTool
    from aeon.core.prompts import TOOL_DESC_SUMMARIZE_TOPIC
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_tool_properties():
    print("Testing tool properties...")
    
    # Use MagicMock to avoid calling the real LLMClient.__init__
    mock_llm_client = MagicMock()
    
    try:
        tool = SummarizeTopicTool(mock_llm_client)
        
        # Verify name
        print(f"Tool name: {tool.name}")
        assert tool.name == "summarize_topic", f"Expected name 'summarize_topic', got '{tool.name}'"
        
        # Verify description
        print(f"Tool description: {tool.description[:100]}...")
        assert tool.description == TOOL_DESC_SUMMARIZE_TOPIC, "Tool description does not match TOOL_DESC_SUMMARIZE_TOPIC"
        
        # Verify underlying model
        print(f"Underlying model: {tool.underlying_model}")
        assert tool.underlying_model == "Tavily", f"Expected model 'Tavily', got '{tool.underlying_model}'"
        
        print("Tool properties verification PASSED.")
    except Exception as e:
        print(f"Tool properties verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_tool_properties()
    print("\nAll tests passed successfully!")