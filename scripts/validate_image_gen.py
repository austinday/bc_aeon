import os
import sys
from aeon.tools.generate_image import GenerateImageTool

def main():
    print("Initializing GenerateImageTool...")
    tool = GenerateImageTool()
    
    prompt = "A high-quality cinematic shot of a futuristic city with neon lights, 8k resolution, highly detailed"
    output_path = "aeon_output/validation_test_image.png"
    
    print(f"Generating image with prompt: {prompt}")
    print(f"Output path: {output_path}")
    
    try:
        result = tool.execute(
            prompt=prompt,
            output_path=output_path,
            width=512,
            height=512
        )
        print(f"Tool result: {result}")
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"Success! Image created at {output_path} (Size: {size} bytes)")
            if size < 1000:
                print("Warning: Image file is suspiciously small.")
                sys.exit(1)
        else:
            print("Error: Output file was not created.")
            sys.exit(1)
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()