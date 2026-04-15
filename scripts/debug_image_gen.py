import os
import sys
import subprocess
from aeon.tools.generate_image import GenerateImageTool

def main():
    print("Starting Debug Image Generation Validation...")
    
    tool = GenerateImageTool()
    prompt = "A high-quality cinematic shot of a futuristic city with neon lights, 8k resolution, highly detailed"
    output_path = "aeon_output/debug_test_image.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Prompt: {prompt}")
    print(f"Output Path: {output_path}")
    
    result = tool.execute(prompt=prompt, output_path=output_path)
    print(f"Tool Result: {result}")
    
    if "Successfully generated" in result and os.path.exists(output_path):
        print("\n✅ SUCCESS: Image generated and saved successfully.")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Image generation failed.")
        print("\n--- Capturing ComfyUI Container Logs ---")
        try:
            logs = subprocess.check_output(["docker", "logs", "aeon_comfyui"], text=True)
            print(logs)
        except subprocess.CalledProcessError as e:
            print(f"Could not retrieve logs: {e}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()