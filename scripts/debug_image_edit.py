import os
import sys
import subprocess
from PIL import Image
from aeon.tools.generate_image import EditImageTool

def create_dummy_image(path):
    print(f"[*] Creating dummy input image at {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Create a simple red square to act as our base image
    img = Image.new('RGB', (512, 512), color='red')
    img.save(path)
    print("[*] Dummy image created.")

def main():
    print("="*60)
    print("Starting Debug Image Edit Validation...")
    print("="*60)
    
    input_path = "aeon_output/debug_edit_input.png"
    output_path = "aeon_output/debug_edit_output.png"
    prompt = "make it blue and snowy, high quality"
    
    # Clean up old output if it exists to ensure we aren't seeing a cached success
    if os.path.exists(output_path):
        os.remove(output_path)
        
    # We need an input image to edit, so we'll generate one on the fly if needed
    if not os.path.exists(input_path):
        create_dummy_image(input_path)
        
    print("\n[*] Initializing EditImageTool...")
    tool = EditImageTool()
    
    print(f"[*] Input Path:  {input_path}")
    print(f"[*] Output Path: {output_path}")
    print(f"[*] Prompt:      '{prompt}'")
    print(f"[*] Denoise:     0.75")
    
    print("\n[*] Executing tool (this may take a few minutes if ComfyUI needs to start)...")
    try:
        result = tool.execute(
            input_path=input_path,
            prompt=prompt,
            output_path=output_path,
            denoise=0.75
        )
        print("\n[*] Tool Execution Result:")
        print(f"    {result}")
        
        if "Successfully edited image" in result and os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"\n✅ SUCCESS: Image edited and saved successfully! (Size: {size} bytes)")
            sys.exit(0)
        else:
            print("\n❌ FAILURE: Image editing failed or output file missing.")
            print("\n--- Capturing ComfyUI Container Logs (Last 50 lines) ---")
            try:
                logs = subprocess.check_output(["docker", "logs", "--tail", "50", "aeon_comfyui"], text=True)
                print(logs)
            except subprocess.CalledProcessError as e:
                print(f"Could not retrieve Docker logs. Is the container running? Error: {e}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ UNEXPECTED EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
