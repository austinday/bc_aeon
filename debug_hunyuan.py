#!/usr/bin/env python3
"""
HunyuanImage Debug Inspector
Runs INSIDE the aeon_comfyui container.
Dumps everything needed to diagnose the img_in.proj.weight KeyError.
"""
import os
import sys
import json
import glob
import subprocess
from pathlib import Path
from collections import Counter, OrderedDict

OUT = '/output/hunyuanDebug.txt'

def w(f, text):
    f.write(text + '\n')
    print(text)

def banner(f, title):
    w(f, '')
    w(f, '=' * 100)
    w(f, f'  {title}')
    w(f, '=' * 100)

def section(f, title):
    w(f, '')
    w(f, '-' * 80)
    w(f, f'  {title}')
    w(f, '-' * 80)

def inspect_safetensors_file(f, filepath):
    section(f, f'SAFETENSORS INSPECTION: {filepath}')
    try:
        from safetensors import safe_open
        st = safe_open(filepath, framework='pt', device='cpu')
        keys = st.keys()
        key_list = sorted(keys)
        w(f, f'Total keys: {len(key_list)}')
        w(f, '')

        # Full key listing with shapes and dtypes
        w(f, 'ALL KEYS (name | shape | dtype):')
        for k in key_list:
            try:
                tensor = st.get_tensor(k)
                w(f, f'  {k:80s} | {str(tensor.shape):30s} | {tensor.dtype}')
            except Exception as e:
                w(f, f'  {k:80s} | ERROR: {e}')

        # Key prefix analysis
        w(f, '')
        w(f, 'KEY PREFIX ANALYSIS (first 2 levels):')
        prefixes = Counter()
        for k in key_list:
            parts = k.split('.')
            prefix = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
            prefixes[prefix] += 1
        for prefix, count in prefixes.most_common():
            w(f, f'  {prefix:60s} : {count} keys')

        # Specifically search for keys the loader might want
        w(f, '')
        w(f, 'SEARCHING FOR KNOWN LOADER KEYS:')
        search_patterns = [
            'img_in', 'txt_in', 'double_blocks', 'single_blocks',
            'final_layer', 'time_in', 'vector_in', 'guidance_in',
            'x_embedder', 'input_blocks', 'output_blocks', 'proj',
            'norm', 'linear', 'dit', 'transformer', 'model.',
            'diffusion_model', 'state_dict'
        ]
        for pat in search_patterns:
            matches = [k for k in key_list if pat in k.lower()]
            w(f, f'  Pattern "{pat}": {len(matches)} matches')
            for m in matches[:5]:
                w(f, f'    -> {m}')
            if len(matches) > 5:
                w(f, f'    ... and {len(matches) - 5} more')

    except ImportError:
        w(f, 'ERROR: safetensors not installed in container')
    except Exception as e:
        w(f, f'ERROR inspecting safetensors: {type(e).__name__}: {e}')

def inspect_wrapper_source(f):
    banner(f, 'HUNYUANVIDEOWRAPPER SOURCE CODE ANALYSIS')

    wrapper_dir = '/opt/ComfyUI/custom_nodes/ComfyUI-HunyuanVideoWrapper'

    # Git info
    section(f, 'GIT VERSION INFO')
    try:
        result = subprocess.run(
            ['git', '-C', wrapper_dir, 'log', '--oneline', '-20'],
            capture_output=True, text=True, timeout=5
        )
        w(f, result.stdout)
    except Exception as e:
        w(f, f'Could not get git log: {e}')

    try:
        result = subprocess.run(
            ['git', '-C', wrapper_dir, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        w(f, f'Current commit: {result.stdout.strip()}')
    except Exception as e:
        w(f, f'Could not get git HEAD: {e}')

    try:
        result = subprocess.run(
            ['git', '-C', wrapper_dir, 'remote', '-v'],
            capture_output=True, text=True, timeout=5
        )
        w(f, f'Remotes:\n{result.stdout}')
    except Exception as e:
        w(f, f'Could not get git remotes: {e}')

    # The critical nodes.py file
    nodes_py = os.path.join(wrapper_dir, 'nodes.py')
    section(f, f'CRITICAL CODE: nodes.py around line 330 (the failing line)')
    if os.path.exists(nodes_py):
        with open(nodes_py, 'r') as src:
            lines = src.readlines()
        w(f, f'Total lines in nodes.py: {len(lines)}')
        # Show lines 300-380 for context around the failure
        start = max(0, 300)
        end = min(len(lines), 400)
        w(f, f'\nLines {start+1}-{end}:')
        for i in range(start, end):
            w(f, f'{i+1:4d} | {lines[i].rstrip()}')
    else:
        w(f, f'ERROR: {nodes_py} not found!')

    # Search for all class definitions and model loading logic
    section(f, 'CLASS DEFINITIONS IN nodes.py')
    if os.path.exists(nodes_py):
        with open(nodes_py, 'r') as src:
            content = src.read()
            lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('class '):
                w(f, f'  Line {i+1}: {line.strip()}')

    # Search for all references to img_in
    section(f, 'ALL REFERENCES TO img_in IN WRAPPER')
    for root, dirs, files in os.walk(wrapper_dir):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r') as src:
                        for i, line in enumerate(src, 1):
                            if 'img_in' in line:
                                rel = os.path.relpath(fpath, wrapper_dir)
                                w(f, f'  {rel}:{i}: {line.rstrip()}')
                except:
                    pass

    # Search for model type detection / key mapping logic
    section(f, 'MODEL TYPE DETECTION LOGIC (searching for key-based model identification)')
    search_terms = ['img_in', 'model_type', 'hunyuan_image', 'HunyuanImage', 'in_channels',
                    'x_embedder', 'key', 'state_dict', 'sd[', 'sd.get']
    if os.path.exists(nodes_py):
        with open(nodes_py, 'r') as src:
            lines = src.readlines()
        for term in search_terms:
            hits = [(i+1, l.rstrip()) for i, l in enumerate(lines) if term in l]
            if hits:
                w(f, f'  \n  Term "{term}" ({len(hits)} hits):')
                for linenum, linetext in hits[:15]:
                    w(f, f'    {linenum:4d} | {linetext}')
                if len(hits) > 15:
                    w(f, f'    ... {len(hits) - 15} more hits')

    # Also check the hyvideo module for model definitions
    section(f, 'HYVIDEO MODULE STRUCTURE')
    hyvideo_dir = os.path.join(wrapper_dir, 'hyvideo')
    if os.path.exists(hyvideo_dir):
        for root, dirs, files in os.walk(hyvideo_dir):
            level = root.replace(hyvideo_dir, '').count(os.sep)
            indent = '  ' * level
            w(f, f'{indent}{os.path.basename(root)}/')
            for fname in sorted(files):
                if fname.endswith('.py'):
                    fpath = os.path.join(root, fname)
                    size = os.path.getsize(fpath)
                    w(f, f'{indent}  {fname} ({size} bytes)')
    else:
        w(f, f'{hyvideo_dir} not found')

    # Check for any model config files or JSON in the wrapper
    section(f, 'CONFIG/JSON FILES IN WRAPPER')
    for root, dirs, files in os.walk(wrapper_dir):
        for fname in files:
            if fname.endswith(('.json', '.yaml', '.yml', '.toml')):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, wrapper_dir)
                size = os.path.getsize(fpath)
                w(f, f'  {rel} ({size} bytes)')
                if size < 10000:
                    try:
                        with open(fpath, 'r') as cfg:
                            w(f, cfg.read())
                    except:
                        pass

def inspect_model_files(f):
    banner(f, 'MODEL FILES ON DISK')

    search_dirs = [
        '/opt/ComfyUI/models/checkpoints',
        '/opt/ComfyUI/models/diffusion_models',
        '/opt/ComfyUI/models/unet',
        '/opt/ComfyUI/models/vae',
        '/opt/ComfyUI/models/clip',
        '/opt/ComfyUI/models/text_encoders',
        '/opt/ComfyUI/models/llm',
        '/opt/ComfyUI/models/LLM',
    ]

    for d in search_dirs:
        section(f, f'DIRECTORY: {d}')
        if not os.path.exists(d):
            w(f, '  (does not exist)')
            continue
        for root, dirs, files in os.walk(d):
            level = root.replace(d, '').count(os.sep)
            if level > 3:
                continue
            indent = '  ' * (level + 1)
            w(f, f'{indent}{os.path.basename(root)}/')
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                try:
                    size = os.path.getsize(fpath)
                    if size > 1024*1024:
                        w(f, f'{indent}  {fname} ({size/1024/1024:.1f} MB)')
                    else:
                        w(f, f'{indent}  {fname} ({size} bytes)')
                except:
                    w(f, f'{indent}  {fname} (?)')

def inspect_hunyuan_int8_repo_metadata(f):
    banner(f, 'HUNYUAN INT8 REPO METADATA')

    # Check for any config/index files in the model dir
    model_dirs = [
        '/opt/ComfyUI/models/checkpoints/hunyuan_image_int8',
        '/opt/ComfyUI/models/diffusion_models/hunyuan_image_int8',
    ]
    for d in model_dirs:
        if not os.path.exists(d):
            continue
        section(f, f'METADATA IN: {d}')
        for fname in os.listdir(d):
            fpath = os.path.join(d, fname)
            if fname.endswith(('.json', '.txt', '.md', '.yaml')):
                w(f, f'\n--- {fname} ---')
                try:
                    with open(fpath, 'r') as mf:
                        content = mf.read()
                        if len(content) < 50000:
                            w(f, content)
                        else:
                            w(f, content[:5000])
                            w(f, f'... (truncated, {len(content)} total chars)')
                except:
                    w(f, '(could not read)')

        # Specifically check the index file for shard structure
        index_file = os.path.join(d, 'model.safetensors.index.json')
        if os.path.exists(index_file):
            section(f, 'SAFETENSORS INDEX FILE ANALYSIS')
            try:
                with open(index_file, 'r') as idx:
                    index_data = json.load(idx)
                metadata = index_data.get('metadata', {})
                w(f, f'Index metadata: {json.dumps(metadata, indent=2)}')
                weight_map = index_data.get('weight_map', {})
                w(f, f'Total weights in index: {len(weight_map)}')
                # Show all key names from the index
                w(f, '\nALL KEYS IN INDEX (weight_map):')
                for k in sorted(weight_map.keys()):
                    w(f, f'  {k} -> {weight_map[k]}')
                # Shard file list
                shards = sorted(set(weight_map.values()))
                w(f, f'\nShard files ({len(shards)}):')
                for s in shards:
                    shard_path = os.path.join(d, s)
                    if os.path.exists(shard_path):
                        sz = os.path.getsize(shard_path)
                        w(f, f'  {s} ({sz/1024/1024:.1f} MB)')
                    else:
                        w(f, f'  {s} (MISSING!)')
            except Exception as e:
                w(f, f'Error reading index: {e}')

def inspect_comfyui_object_info(f):
    banner(f, 'COMFYUI NODE REGISTRY (from /object_info API)')
    try:
        import urllib.request
        req = urllib.request.Request('http://localhost:8188/object_info')
        resp = urllib.request.urlopen(req, timeout=10)
        all_nodes = json.loads(resp.read().decode('utf-8'))

        # Filter to relevant nodes
        keywords = ['hunyuan', 'hyvideo', 'gguf', 'image', 'model', 'loader', 'unet', 'dit']
        relevant = {}
        for name, info in all_nodes.items():
            nl = name.lower()
            if any(kw in nl for kw in keywords):
                relevant[name] = info

        section(f, f'RELEVANT NODES ({len(relevant)} of {len(all_nodes)} total)')
        for name, info in sorted(relevant.items()):
            w(f, f'\n  NODE: {name}')
            w(f, f'  Category: {info.get("category", "?")}')
            w(f, f'  Description: {info.get("description", "?")}')
            req_inputs = info.get('input', {}).get('required', {})
            opt_inputs = info.get('input', {}).get('optional', {})
            if req_inputs:
                w(f, f'  Required Inputs:')
                for iname, ispec in req_inputs.items():
                    w(f, f'    {iname}: {json.dumps(ispec, default=str)[:200]}')
            if opt_inputs:
                w(f, f'  Optional Inputs:')
                for iname, ispec in opt_inputs.items():
                    w(f, f'    {iname}: {json.dumps(ispec, default=str)[:200]}')
            output_types = info.get('output', [])
            w(f, f'  Outputs: {output_types}')

    except Exception as e:
        w(f, f'Could not query /object_info (is ComfyUI running?): {e}')
        w(f, 'Falling back to source code inspection only.')

def inspect_individual_shards(f):
    """Check if individual shard files have different key structure than merged."""
    banner(f, 'INDIVIDUAL SHARD KEY COMPARISON')
    model_dirs = [
        '/opt/ComfyUI/models/checkpoints/hunyuan_image_int8',
        '/opt/ComfyUI/models/diffusion_models/hunyuan_image_int8',
    ]
    try:
        from safetensors import safe_open
        for d in model_dirs:
            if not os.path.exists(d):
                continue
            shards = sorted(glob.glob(os.path.join(d, 'model-*.safetensors')))
            if not shards:
                shards = sorted(glob.glob(os.path.join(d, '*.safetensors')))
            for shard in shards[:3]:  # Check first 3 shards
                section(f, f'SHARD: {os.path.basename(shard)}')
                try:
                    st = safe_open(shard, framework='pt', device='cpu')
                    keys = sorted(st.keys())
                    w(f, f'  Keys in this shard: {len(keys)}')
                    w(f, f'  First 20 keys:')
                    for k in keys[:20]:
                        tensor = st.get_tensor(k)
                        w(f, f'    {k} | {tensor.shape} | {tensor.dtype}')
                    if len(keys) > 20:
                        w(f, f'  ... and {len(keys) - 20} more')
                except Exception as e:
                    w(f, f'  Error: {e}')
    except ImportError:
        w(f, 'safetensors not available')

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        banner(f, 'HUNYUAN IMAGE DEBUG REPORT')
        w(f, f'Generated by debug_hunyuan.py')
        w(f, f'Python: {sys.version}')
        w(f, '')

        # 1. Model files on disk
        inspect_model_files(f)

        # 2. INT8 repo metadata (config.json, index.json, README, etc.)
        inspect_hunyuan_int8_repo_metadata(f)

        # 3. Merged safetensors key inspection
        merged_paths = [
            '/opt/ComfyUI/models/checkpoints/hunyuan_image_int8/hunyuan_image_merged.safetensors',
            '/opt/ComfyUI/models/diffusion_models/hunyuan_image_int8/hunyuan_image_merged.safetensors',
        ]
        for mp in merged_paths:
            if os.path.exists(mp):
                inspect_safetensors_file(f, mp)

        # 4. Individual shard comparison
        inspect_individual_shards(f)

        # 5. Wrapper source code analysis
        inspect_wrapper_source(f)

        # 6. ComfyUI node registry (if API available)
        inspect_comfyui_object_info(f)

        banner(f, 'END OF DEBUG REPORT')

    print(f'\nDebug report written to {OUT}')

if __name__ == '__main__':
    main()
