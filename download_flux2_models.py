import os
import sys
from huggingface_hub import hf_hub_download
import shutil
import stat

def download_model(repo_id, filename, target_path, expected_size):
    cache_dir = '/tmp/hf_cache'
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    print(f'Downloaded {filename} from {repo_id} to {local_path}')

    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    os.chmod(target_dir, 0o777)

    if os.path.exists(target_path):
        current_size = os.path.getsize(target_path)
        if current_size == expected_size:
            print(f'{target_path} already correct size ({current_size}B), skipping.')
            return True
        else:
            print(f'{target_path} size mismatch ({current_size} vs {expected_size}B), redownloading.')
            os.remove(target_path)

    try:
        shutil.copy2(local_path, target_path)
        os.chmod(target_path, 0o666)
        actual_size = os.path.getsize(target_path)
        print(f'Copied to {target_path}, size: {actual_size}B')
        if actual_size == expected_size:
            return True
        else:
            print(f'Size mismatch after copy: {actual_size} vs {expected_size}B')
            return False
    except Exception as e:
        print(f'copy2 failed: {e}, fallback copyfileobj')
        with open(local_path, 'rb') as src, open(target_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        os.chmod(target_path, 0o666)
        actual_size = os.path.getsize(target_path)
        print(f'Fallback copy to {target_path}, size: {actual_size}B')
        return actual_size == expected_size

models = [
    {
        'repo_id': 'Comfy-Org/flux2-dev',
        'filename': 'split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors',
        'target': 'aeon_models/comfyui_models/clip/mistral_3_small_flux2_fp8.safetensors',
        'size': 2684354560  # ~2.5GB
    },
    {
        'repo_id': 'Comfy-Org/flux2-dev',
        'filename': 'split_files/diffusion_models/flux2_dev_fp8mixed.safetensors',
        'target': 'aeon_models/comfyui_models/unet/flux2_dev_fp8mixed.safetensors',
        'size': 12900000000  # ~12GB
    },
    {
        'repo_id': 'Comfy-Org/flux2-dev',
        'filename': 'split_files/vae/flux2-vae.safetensors',
        'target': 'aeon_models/comfyui_models/vae/flux2-vae.safetensors',
        'size': 335544320  # ~320MB
    },
    {
        'repo_id': 'Lakonik/pi-FLUX.2',
        'filename': 'gmflux2_k8_piid_4step/diffusion_pytorch_model.safetensors',
        'target': 'aeon_models/comfyui_models/loras/gmflux2_k8_piid_4step.safetensors',
        'size': 1468006400  # ~1.4GB
    }
]

if 'HF_TOKEN' not in os.environ:
    print('ERROR: HF_TOKEN env var required.', file=sys.stderr)
    sys.exit(1)

os.environ['HF_TOKEN'] = os.environ['HF_TOKEN']

success = True
for model in models:
    if not download_model(model['repo_id'], model['filename'], model['target'], model['size']):
        success = False

if success:
    print('All Flux.2 piFlow models downloaded and verified successfully.')
else:
    print('Some models failed to download/verify.', file=sys.stderr)
    sys.exit(1)