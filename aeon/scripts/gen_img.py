import argparse
import torch
import os
import time
import sys
import json
import re
from threading import Thread, Lock
from queue import Queue, Empty

if not os.path.exists("/.dockerenv"):
    print("CRITICAL: This script must be run inside the container.")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def cprint(text, color="white"):
    colors = {'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m', 'magenta': '\033[95m', 'cyan': '\033[96m'}
    print(f"{colors.get(color, '')}{text}\033[0m")

try:
    from diffusers import FluxPipeline, AutoPipelineForText2Image
except ImportError:
    cprint("Dependencies missing.", "red")
    sys.exit(1)

def get_dimensions(aspect_ratio, is_flux):
    mapping = {
        "1:1": (1024, 1024) if is_flux else (512, 512),
        "16:9": (1344, 768) if is_flux else (768, 448),
        "9:16": (768, 1344) if is_flux else (448, 768),
        "4:3": (1152, 896) if is_flux else (640, 480)
    }
    return mapping.get(aspect_ratio, mapping["1:1"])

def auto_select_model(base_path):
    # Detect VRAM to choose between Flux (High) and DreamShaper (Efficient)
    if not torch.cuda.is_available():
        return "dreamshaper", os.path.join(base_path, "dreamshaper_8")
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cprint(f"System VRAM detected: {total_vram:.1f} GB", "cyan")
    
    # Flux needs ~12-16GB to run comfortably in bfloat16
    if total_vram > 15.0:
        cprint("Selecting SOTA Model: Flux.1 Schnell", "magenta")
        return "flux", os.path.join(base_path, "flux_schnell")
    else:
        cprint("Selecting Efficient Model: DreamShaper 8", "magenta")
        return "dreamshaper", os.path.join(base_path, "dreamshaper_8")

def load_pipe(m_type, path, dev_id):
    device = "cpu" if dev_id == "cpu" else f"cuda:{dev_id}"
    dtype = torch.bfloat16 if m_type == 'flux' else torch.float16
    if device == "cpu": dtype = torch.float32
    
    if m_type == 'flux':
        pipe = FluxPipeline.from_pretrained(path, torch_dtype=dtype, local_files_only=True)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dtype, local_files_only=True)
    
    pipe.to(device)
    return pipe

def worker_thread(dev_id, m_type, path, task_queue, out_dir, pbar, pbar_lock):
    try:
        pipe = load_pipe(m_type, path, dev_id)
        steps = 4 if m_type == 'flux' else 25
        
        while True:
            try:
                prompt, prefix, i_idx, ratio = task_queue.get_nowait()
            except Empty: break
                
            w, h = get_dimensions(ratio, m_type == 'flux')
            fname = f"{out_dir}/{prefix}_{i_idx}.png"
            
            cprint(f"[Device {dev_id}] Drawing: {prefix} ({w}x{h})...", "green")
            
            if m_type == 'flux':
                img = pipe(prompt, width=w, height=h, num_inference_steps=steps, guidance_scale=0.0, max_sequence_length=256).images[0]
            else:
                img = pipe(prompt, width=w, height=h, num_inference_steps=steps, guidance_scale=7.5).images[0]
            
            img.save(fname)
            task_queue.task_done()
            if pbar: 
                with pbar_lock: pbar.update(1)
    except Exception as e:
        cprint(f"Worker failed: {e}", "red")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--requests_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    m_type, m_path = auto_select_model(args.models_dir)
    
    with open(args.requests_file, 'r') as f: requests = json.load(f)
    task_queue = Queue()
    for req in requests:
        p, c, r = req.get('prompt', ''), req.get('count', 1), req.get('aspect_ratio', '1:1')
        name = req.get('name') or "gen"
        for i in range(c): task_queue.put((p, name, i, r))

    devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else ['cpu']
    pbar, pbar_lock = (tqdm(total=task_queue.qsize(), desc="Vision Progress"), Lock()) if tqdm else (None, None)
    
    threads = [Thread(target=worker_thread, args=(d, m_type, m_path, task_queue, args.output_dir, pbar, pbar_lock)) for d in devs]
    for t in threads: t.start()
    for t in threads: t.join()
    if pbar: pbar.close()

if __name__ == "__main__":
    main()