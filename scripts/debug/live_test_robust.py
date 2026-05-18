import os
import json
import unittest
from unittest.mock import patch, MagicMock
from aeon.tools.generate_video import GenerateVideoTool

class TestGenerateVideoTool(unittest.TestCase):
    def setUp(self):
        self.tool = GenerateVideoTool()
        # Mock the output directory to avoid actual file system clutter during logic tests
        self.tool.output_dir = "aeon_output/debug/test_out"
        os.makedirs(self.tool.output_dir, exist_ok=True)

    @patch('requests.post')
    @patch('requests.get')
    def test_text_to_video_tiktok_optimization(self, mock_get, mock_post):
        """Test short prompt optimization and tiktok aspect ratio."""
        # Mock API responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"prompt_id": "test_id_123"}
        
        mock_get.return_value.json.return_value = {"test_id_123": {"status": "completed"}}
        
        # Create a dummy file to simulate ComfyUI output
        dummy_video = os.path.join(self.tool.output_dir, "test_video.mp4")
        with open(dummy_video, "w") as f:
            f.write("dummy content")

        # Execute tool: short prompt + tiktok preset
        result = self.tool.execute(
            mode="text_to_video",
            prompt="a cat",
            output_path="aeon_output/debug/result_tiktok.mp4",
            width="tiktok",
            height="tiktok", # Tool handles this via the preset map
            frames=33
        )

        # Verify prompt optimization happened
        # The tool should have expanded "a cat"
        args, kwargs = mock_post.call_args
        payload = args[1] if 'json' not in kwargs else kwargs['json']
        sent_prompt = payload['prompt']['4']['inputs']['text']
        
        self.assertIn("high quality", sent_prompt)
        self.assertIn("a cat", sent_prompt)
        
        # Verify aspect ratio preset (tiktok: 512, 896)
        self.assertEqual(payload['prompt']['11']['inputs']['width'], 512)
        self.assertEqual(payload['prompt']['11']['inputs']['height'], 896)
        
        print("Test 1 (TikTok + Optimization) PASSED")

    @patch('requests.post')
    @patch('requests.get')
    def test_cinematic_aspect_ratio(self, mock_get, mock_post):
        """Test cinematic aspect ratio preset."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"prompt_id": "test_id_456"}
        mock_get.return_value.json.return_value = {"test_id_456": {"status": "completed"}}
        
        dummy_video = os.path.join(self.tool.output_dir, "test_video_cin.mp4")
        with open(dummy_video, "w") as f:
            f.write("dummy content")

        self.tool.execute(
            mode="text_to_video",
            prompt="cinematic landscape",
            output_path="aeon_output/debug/result_cin.mp4",
            width="cinematic",
            frames=33
        )

        args, kwargs = mock_post.call_args
        payload = args[1] if 'json' not in kwargs else kwargs['json']
        
        # Cinematic: (1280, 544)
        self.assertEqual(payload['prompt']['11']['inputs']['width'], 1280)
        self.assertEqual(payload['prompt']['11']['inputs']['height'], 544)
        print("Test 2 (Cinematic Aspect Ratio) PASSED")

if __name__ == "__main__":
    unittest.main()