import sys
import os

# Ensure the 'aeon' package is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from aeon.tools.generate_image import GenerateImageTool

def main():
    print("Initializing GenerateImageTool...")
    tool = GenerateImageTool()
    
    prompt = "A highly detailed photorealistic portrait of a young woman with sharp facial features, realistic skin texture, soft natural lighting, professional photography, 8k UHD"
    print(f"Executing with prompt: '{prompt}'")
    
    try:
        result = tool.execute(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            width=1024,
            height=1024,
            steps=8,  # Recommended for Distil INT8
            cfg_scale=2.5,  # Recommended for Distil
            flow_shift=7.0,
            seed=-1
        )
        print("\n=== GENERATION RESULT ===")
        print(result)
    except Exception as e:
        import traceback
        print("\n=== GENERATION FAILED ===")
        traceback.print_exc()

if __name__ == "__main__":
    main()
