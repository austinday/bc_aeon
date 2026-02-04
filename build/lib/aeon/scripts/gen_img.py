import argparse
import json
import os
import torch
from diffusers import DiffusionPipeline
from PIL import Image

def load_pipeline(models_dir):
    model_path = os.path.join(models_dir, "HunyuanImage-3.0")
    print(f"Loading Hunyuan model from: {model_path}")
    
    try:
        # Hunyuan usually requires standard pipeline loading or specific HunyuanDiTPipeline
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe.to("cuda")
        return pipe
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

def process_requests(requests_file, output_dir, pipe):
    with open(requests_file, 'r') as f:
        requests = json.load(f)

    for i, req in enumerate(requests):
        prompt = req.get("prompt")
        name = req.get("name") or f"image_{i}"
        count = req.get("count", 1)
        init_img_path = req.get("init_image_path")
        
        print(f"Processing: {name} | Prompt: {prompt[:50]}...")

        for j in range(count):
            # Basic Img2Img or Txt2Img logic
            if init_img_path and os.path.exists(init_img_path):
                # TODO: Implement Img2Img pipeline logic if model supports it
                # For now, default to Txt2Img as fallback or load specific Img2Img pipe
                print("Warning: Img2Img requested but Txt2Img pipeline active. Proceeding with Txt2Img.")
            
            image = pipe(prompt=prompt, num_inference_steps=25).images[0]
            
            save_path = os.path.join(output_dir, f"{name}_{j}.png")
            image.save(save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--requests_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    pipe = load_pipeline(args.models_dir)
    process_requests(args.requests_file, args.output_dir, pipe)
