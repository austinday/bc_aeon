import os
from huggingface_hub import hf_hub_download

base_dir = '/app/aeon_models/comfyui_models'
models = [
    {
        'repo_id': 'Comfy-Org/flux2-dev',
        'filename': 'split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors',
        'target_dir': f'{base_dir}/clip',
        'target_file': 'mistral_3_small_flux2_fp8.safetensors',
        'expected_size_mb': 4500
    },
    {
        'repo_id': 'Comfy-Org/flux2-dev',
        'filename': 'split_files/diffusion_models/flux2_dev_fp8mixed.safetensors',
        'target_dir': f'{base_dir}/unet',
        'target_file': 'flux2_dev_fp8mixed.safetensors',
        'expected_size_mb': 12000
    },
    {
        'repo_id': 'Comfy-Org/flux2-dev',
        'filename': 'split_files/vae/flux2-vae.safetensors',
        'target_dir': f'{base_dir}/vae',
        'target_file': 'flux2-vae.safetensors',
        'expected_size_mb': 336
    },
    {
        'repo_id': 'Lakonik/pi-FLUX.2',
        'filename': 'gmflux2_k8_piid_4step/diffusion_pytorch_model.safetensors',
        'target_dir': f'{base_dir}/loras',
        'target_file': 'gmflux2_k8_piid_4step.safetensors',
        'expected_size_mb': 50
    }
]

os.makedirs(base_dir, exist_ok=True)
for model in models:
    target_path = os.path.join(model['target_dir'], model['target_file'])
    os.makedirs(model['target_dir'], exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) / (1024*1024) > model['expected_size_mb'] * 0.9:
        print(f'Skipping {model["target_file"]}: exists and size matches (~{model["expected_size_mb"]}MB).')
        continue
    print(f'Downloading {model["target_file"]} from {model["repo_id"]}...')
    hf_hub_download(
        repo_id=model['repo_id'],
        filename=model['filename'],
        local_dir=model['target_dir'],
        local_dir_target=model['target_file']
    )
    print(f'Downloaded {model["target_file"]} ({os.path.getsize(target_path)/(1024*1024):.1f}MB)')
print('All Flux.2 piFlow models ready!')