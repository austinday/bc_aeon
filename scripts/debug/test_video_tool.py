import sys
import os

# Add the root directory to sys.path to allow importing from aeon.tools
sys.path.append(os.getcwd())

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("Successfully imported GenerateVideoTool")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_logic():
    tool = GenerateVideoTool()
    
    # Test Prompt Optimization
    short_prompt = "A cat"
    optimized = tool._optimize_prompt(short_prompt)
    print(f"Short prompt: {short_prompt} -> Optimized: {optimized}")
    assert "high quality" in optimized or "detailed" in optimized
    
    # Test Aspect Ratio Presets
    # Test 'tiktok'
    try:
        # We mock the execute method's internal logic for aspect ratios
        # since we don't want to actually call ComfyUI during a logic test
        # but we can check if the logic in execute handles it.
        # For a pure logic test, we can just check the dictionary.
        print(f"Testing 'tiktok' preset: {tool.aspect_ratios['tiktok']}")
        assert tool.aspect_ratios['tiktok'] == (512, 896)
        
        print(f"Testing 'cinematic' preset: {tool.aspect_ratios['cinematic']}")
        assert tool.aspect_ratios['cinematic'] == (1280, 544)
    except KeyError as e:
        print(f"Preset missing: {e}")
        sys.exit(1)

    print("Logic tests passed successfully!")

if __name__ == "__main__":
    test_logic()