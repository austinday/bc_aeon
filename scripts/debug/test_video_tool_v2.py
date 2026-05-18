import sys
import os

# Force use of local source directory
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd())))

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("Successfully imported GenerateVideoTool from local source.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_logic():
    tool = GenerateVideoTool()
    
    # Test 1: Prompt Optimization
    short_prompt = "a cat"
    optimized = tool._optimize_prompt(short_prompt)
    print(f"Short Prompt: {short_prompt} -> Optimized: {optimized}")
    assert "high quality" in optimized or "detailed" in optimized
    
    # Test 2: Aspect Ratio Presets
    # Test 'tiktok' preset
    res = tool.execute(
        mode="text_to_video", 
        prompt="test", 
        output_path="test.mp4", 
        width="tiktok", 
        frames=1
    )
    # Since we don't have a running ComfyUI in this test, we expect it to fail 
    # at the request stage, but we want to see if it gets past the preset logic.
    if "Error during video generation" in res and "Connection refused" in res:
        print("Preset logic passed (reached API call stage).")
    elif "Error during video generation" in res and "AttributeError" in res:
        print(f"FAILED: Still seeing AttributeError: {res}")
        sys.exit(1)
    else:
        print(f"Result: {res}")

if __name__ == "__main__":
    test_logic()
    print("Validation successful!")