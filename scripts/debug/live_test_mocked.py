import sys
import os
from unittest.mock import MagicMock, patch

# Add the root directory to sys.path to import aeon
sys.path.append(os.getcwd())

from aeon.tools.generate_video import GenerateVideoTool

def test_tool_logic():
    print("Testing GenerateVideoTool logic with mocked API...")
    tool = GenerateVideoTool()
    
    # Test Case 1: Text-to-Video with Aspect Ratio Preset 'tiktok'
    # We mock requests.post and requests.get to avoid actual network calls
    with patch('requests.post') as mock_post, patch('requests.get') as mock_get:
        # Mock the API responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"prompt_id": "12345"}
        mock_get.return_value.json.return_value = {"12345": {"status": "completed"}}
        
        # Mock file system for output
        with patch('os.listdir', return_value=["test_video.mp4"]), \
             patch('os.path.getmtime', return_value=100), \
             patch('shutil.copy') as mock_copy:
            
            print("\nRunning Test 1: TikTok aspect ratio and short prompt optimization...")
            result = tool.execute(
                mode="text_to_video",
                prompt="A cat dancing",
                output_path="aeon_output/test_tiktok.mp4",
                width="tiktok"
            )
            print(f"Result: {result}")
            
            # Verify the prompt was optimized
            args, kwargs = mock_post.call_args
            payload = args[0] if args else kwargs.get('json', {})
            if isinstance(payload, dict) and 'prompt' in payload:
                workflow = payload['prompt']
                prompt_text = workflow['4']['inputs']['text']
                print(f"Sent Prompt: {prompt_text}")
                assert "high quality" in prompt_text
                
                # Verify aspect ratio
                width = workflow['11']['inputs']['width']
                height = workflow['11']['inputs']['height']
                print(f"Sent Dimensions: {width}x{height}")
                assert width == 512 and height == 896
            else:
                print("Payload structure unexpected")
                sys.exit(1)

    print("\nLogic verification successful!")

if __name__ == "__main__":
    try:
        test_tool_logic()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)