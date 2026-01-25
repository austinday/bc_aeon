import argparse
import torch
import os
import sys
from PIL import Image
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

# Force offline
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

def get_system_vram():
    if not torch.cuda.is_available(): return 0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)

def vision_scheduler(base_path):
    vram = get_system_vram()
    print(f'[DEBUG] Detected VRAM: {vram:.2f} GB')
    
    # Tier 3: 96GB GPU -> Qwen2-VL-72B
    if vram >= 75.0:
        return 'vision_large', os.path.join(base_path, 'vision_large'), 'qwen2_72b'
    
    # Tier 2: 24GB GPU -> InternVL2-26B (Requires 4-bit to fit in 24GB)
    elif vram >= 20.0:
        return 'vision_med', os.path.join(base_path, 'vision_med'), 'internvl2_26b'
    
    # Tier 1: 6GB GPU -> Qwen2-VL-2B (Fits in BF16/FP16)
    else:
        return 'vision_small', os.path.join(base_path, 'vision_small'), 'qwen2_2b'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', required=True)
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--prompt', required=True)
    args = parser.parse_args()

    m_key, m_path, m_family = vision_scheduler(args.models_dir)

    if not os.path.exists(m_path):
        print(f'CRITICAL: Model directory {m_path} not found.')
        sys.exit(1)

    # Add path to sys.path for trust_remote_code
    sys.path.append(m_path)

    try:
        image = Image.open(args.image_path).convert('RGB')
    except Exception as e:
        print(f'Error opening image: {e}')
        sys.exit(1)

    # CONFIGURATION FOR MODELS
    load_kwargs = {
        'trust_remote_code': True,
        'device_map': 'auto',
        'local_files_only': True
    }

    # Quantization Logic
    if m_family == 'internvl2_26b' or m_family == 'qwen2_72b':
        print('[DEBUG] Applying 4-bit Quantization (NF4)...')
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4'
        )
    else:
        # Small model fits in native BF16
        load_kwargs['torch_dtype'] = torch.bfloat16

    try:
        print(f'[DEBUG] Loading {m_key}...')
        
        # INSTANTIATE MODEL WITH CORRECT CLASS
        if 'qwen2' in m_family:
            # Qwen2-VL requires specific conditional generation class
            model = Qwen2VLForConditionalGeneration.from_pretrained(m_path, **load_kwargs)
        elif 'internvl2' in m_family:
            # InternVL2 relies on AutoModel with trust_remote_code
            model = AutoModel.from_pretrained(m_path, **load_kwargs)
        else:
            # Fallback for generic models (should not be hit with current registry)
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(m_path, **load_kwargs)

        processor = AutoProcessor.from_pretrained(m_path, trust_remote_code=True, local_files_only=True)

        # PROMPT CONSTRUCTION
        # Standardized Chat Template format for VLMs
        messages = [
            {
                'role': 'user', 
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': args.prompt}
                ]
            }
        ]

        # Prepare inputs using chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        # Qwen2-VL processor handles images automatically from the messages list in newer versions,
        # but explicit passing is safer for compatibility.
        image_inputs = [image]
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
        # Decode (strip input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        print('--- VISION ANALYSIS START ---')
        print(output_text)
        print('--- VISION ANALYSIS END ---')

    except Exception as e:
        print('--- ERROR START ---')
        print(f'Inference Failed: {e}')
        import traceback
        traceback.print_exc()
        print('--- ERROR END ---')
        sys.exit(1)

if __name__ == '__main__':
    main()