import os
import sys
from unittest.mock import MagicMock, patch

# Add the project root to the front of sys.path to ensure the local version is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from aeon.tools.generate_video import GenerateVideoTool

def test_paths():
    print("Testing GenerateVideoTool path independence...")
    
    # Instantiate the tool
    tool = GenerateVideoTool()
    
    # Verify that internal directories are based on AEON_HOME, not current working directory
    # AEON_HOME defaults to /home/aday/.aeon
    expected_home = os.environ.get("AEON_HOME", "/home/aday/.aeon")
    assert tool.output_dir.startswith(expected_home), f"output_dir should start with {expected_home}, got {tool.output_dir}"
    assert tool.input_dir.startswith(expected_home), f"input_dir should start with {expected_home}, got {tool.input_dir}"
    assert tool.debug_dir.startswith(expected_home), f"debug_dir should start with {expected_home}, got {tool.debug_dir}"
    
    print("Internal directories are correctly based on AEON_HOME.")

    # Mock the external dependencies to test the 'execute' logic without needing ComfyUI
    with patch.object(tool, '_manage_registry'), \
         patch.object(tool, '_generate_single_chunk') as mock_gen, \
         patch.object(tool, '_extract_last_frame') as mock_extract, \
         patch.object(tool, '_concatenate_videos') as mock_concat:
        
        # Change directory to /tmp to simulate running from outside the workspace
        original_cwd = os.getcwd()
        os.chdir("/tmp")
        try:
            test_output = "/tmp/test_video.mp4"
            # Generate a long video (more than max_chunk_frames) to trigger recursive logic
            tool.execute(
                mode="text_to_video", 
                prompt="A test prompt", 
                output_path=test_output, 
                frames=100 # 100 > 33, will trigger chunks
            )
            
            # Verify that chunk outputs are created in debug_dir
            # The first chunk should be in tool.debug_dir
            first_chunk_call = mock_gen.call_args_list[0]
            chunk_output_path = first_chunk_call[0][2]
            assert chunk_output_path.startswith(tool.debug_dir), f"Chunk output should be in {tool.debug_dir}, got {chunk_output_path}"
            
            # Verify that extract_last_frame is called with paths in debug_dir
            extract_call = mock_extract.call_args_list[0]
            video_path, image_path = extract_call[0]
            assert video_path.startswith(tool.debug_dir), f"Extract video path should be in {tool.debug_dir}, got {video_path}"
            assert image_path.startswith(tool.debug_dir), f"Extract image path should be in {tool.debug_dir}, got {image_path}"
            
            # Verify that concatenate_videos is called with the final output path
            concat_call = mock_concat.call_args_list[0]
            chunks, final_output = concat_call[0]
            assert os.path.abspath(final_output) == os.path.abspath(test_output), f"Final output path mismatch: {final_output} vs {test_output}"
            for chunk in chunks:
                assert chunk.startswith(tool.debug_dir), f"Concatenated chunk should be in {tool.debug_dir}, got {chunk}"
            
            print("Path logic for recursive generation is correct and independent of CWD.")
            
        finally:
            os.chdir(original_cwd)

    print("All path tests passed!")

if __name__ == "__main__":
    try:
        test_paths()
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)