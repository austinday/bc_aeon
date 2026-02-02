import platform
import psutil
import os
import re
from collections import Counter
from datetime import datetime

# --- Configuration ---
# Adjusted for 128k Context Window
# We allow the tree to take up to 100k tokens, leaving ~28k for conversation/history.
MAX_TREE_TOKENS = 100000 
CHARS_PER_TOKEN = 4
MAX_OUTPUT_CHARS = MAX_TREE_TOKENS * CHARS_PER_TOKEN

def get_system_specs():
    """Gathers and formats system hardware and software information."""
    specs = []

    # OS Info
    uname = platform.uname()
    specs.append(f"Operating System: {uname.system} {uname.release} ({uname.version})")

    # CPU Info
    specs.append(f"CPU: {psutil.cpu_count(logical=True)} Logical")

    # Memory Info
    svmem = psutil.virtual_memory()
    specs.append(f"Memory: {svmem.total / (1024**3):.2f} GB")

    # GPU Info
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlSystemGetDriverVersion, NVMLError
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        if gpu_count > 0:
            driver_version_raw = nvmlSystemGetDriverVersion()
            driver_version = driver_version_raw.decode('utf-8') if isinstance(driver_version_raw, bytes) else driver_version_raw
            specs.append(f"NVIDIA Driver Version: {driver_version}")
            for i in range(gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                gpu_name_raw = nvmlDeviceGetName(handle)
                gpu_name = gpu_name_raw.decode('utf-8') if isinstance(gpu_name_raw, bytes) else gpu_name_raw
                specs.append(f"GPU {i}: {gpu_name}")
        else:
            specs.append("GPU: No NVIDIA GPUs detected.")
    except (ImportError, NVMLError):
        specs.append("GPU: NVIDIA driver/pynvml not installed or failed to initialize.")

    return "\n".join(specs)

def _analyze_file_list_smart(files):
    """
    Intelligently compresses a list of files.
    Returns: (list_of_files_to_display, summary_string_or_None)
    """
    total_files = len(files)
    
    # 1. Threshold Check
    if total_files <= 100:
        return files, None

    # 2. Sequential/Pattern Check
    # Regex to capture: Prefix + Number + Suffix (extension)
    # e.g., "variant_" + "001" + ".pdb"
    pattern_regex = re.compile(r'^(.*?)(\d+)(\.[^.]+)$')
    
    # Group files by their "pattern signature" (prefix + suffix)
    patterns = Counter()
    for f in files:
        match = pattern_regex.match(f)
        if match:
            # Signature is prefix + suffix (e.g., "variant_|.pdb")
            signature = (match.group(1), match.group(3))
            patterns[signature] += 1
            
    # Check if a single pattern dominates (e.g., > 50% of files)
    if patterns:
        most_common_sig, count = patterns.most_common(1)[0]
        if count > total_files * 0.5:
            prefix, suffix = most_common_sig
            # Find the first 3 files that match this pattern
            example_files = [f for f in files if f.startswith(prefix) and f.endswith(suffix)][:3]
            
            summary_msg = (f"... [{total_files - 3} more files. "
                           f"Detected sequential pattern: '{prefix}<N>{suffix}' ({count} files)]")
            return example_files, summary_msg

    # 3. Extension Summary (Fallback)
    # Just show first 10, then summarize by extension
    first_few = files[:10]
    remaining_files = files[10:]
    
    ext_counts = Counter()
    for f in remaining_files:
        _, ext = os.path.splitext(f)
        ext_counts[ext] += 1
    
    # specific format: "X .pdb, Y .txt"
    ext_summary = ", ".join([f"{count} {ext if ext else 'no-ext'}" for ext, count in ext_counts.items()])
    summary_msg = f"... [{len(remaining_files)} more files. Summary: {ext_summary}]"
    
    return first_few, summary_msg

def _generate_tree_internal(startpath, strict_file_limit=None, strict_dir_limit=None):
    """
    Internal generator that builds the tree string.
    strict_file_limit: If int, hard truncate files to this number (bypassing smart logic).
    strict_dir_limit: If int, hard truncate subdirectories to this number.
    """
    tree_lines = []
    tree_lines.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for root, dirs, files in os.walk(startpath, topdown=True):
        # Filter hidden
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        files = [f for f in files if not f.startswith('.')]
        files.sort()
        dirs.sort()

        # --- Strict Directory Truncation ---
        if strict_dir_limit is not None and len(dirs) > strict_dir_limit:
            removed_count = len(dirs) - strict_dir_limit
            dirs[:] = dirs[:strict_dir_limit] # Modify in-place to stop recursion
            # We add a placeholder in the tree display logic below, or we can add a fake dir name
            # But usually, the visual indication comes from the file listing part. 
            # Ideally, we just append a note to the tree lines.
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        
        if root != '.':
            tree_lines.append(f'{indent}{os.path.basename(root)}/')
            # If we truncated dirs, note it here
            if strict_dir_limit is not None and len(dirs) == strict_dir_limit: 
                 # This check is imperfect because we don't know if we *actually* removed any without checking original len
                 # But valid for a simple logic.
                 pass 

        sub_indent = ' ' * 4 * (level + 1)
        
        # --- File List Logic ---
        files_to_display = []
        summary_note = None

        if strict_file_limit is not None:
            # Strict Mode (Panic Mode)
            if len(files) > strict_file_limit:
                files_to_display = files[:strict_file_limit]
                summary_note = f"... [{len(files) - strict_file_limit} files omitted due to context size limit]"
            else:
                files_to_display = files
        else:
            # Smart Mode (Standard)
            files_to_display, summary_note = _analyze_file_list_smart(files)

        # Print Files
        for f in files_to_display:
            try:
                full_path = os.path.join(root, f)
                stat = os.stat(full_path)
                size_bytes = stat.st_size
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                tree_lines.append(f'{sub_indent}{f} (Size: {size_bytes}B, Modified: {mod_time})')
            except OSError:
                tree_lines.append(f'{sub_indent}{f} (unable to get info)')
        
        if summary_note:
            tree_lines.append(f'{sub_indent}{summary_note}')
            
        # If strict dir limit was applied, add a note at the bottom of the dir listing
        # (This is tricky in os.walk, so we do it by checking if we are about to descend)
        
    return "\n".join(tree_lines)

def get_directory_tree_str(startpath='.'):
    """
    Generates a directory tree string with 3 levels of safeguards against context explosion.
    """
    # 1. Attempt Standard Smart Generation
    output = _generate_tree_internal(startpath)
    
    if len(output) < MAX_OUTPUT_CHARS:
        return output
    
    # 2. Level 1 Safeguard: Strict File Limit (Max 10 files per dir)
    # The previous output was too big (likely 100k+ tokens). Truncate files aggressively.
    warning_header = (f"!!! WARNING: Directory tree too large (> {MAX_TREE_TOKENS} tokens). "
                      "Applied SAFEGUARD 1: Truncating to max 10 files per directory.\n")
    output = warning_header + _generate_tree_internal(startpath, strict_file_limit=10)
    
    if len(output) < MAX_OUTPUT_CHARS:
        return output

    # 3. Level 2 Safeguard: Strict Directory Limit (Max 10 dirs per dir)
    # Still too big. This implies a massive folder structure (width/depth).
    warning_header = (f"!!! WARNING: Directory tree EXTREMELY large. "
                      "Applied SAFEGUARD 2: Truncating to max 10 files AND max 10 subdirectories per directory.\n")
    output = warning_header + _generate_tree_internal(startpath, strict_file_limit=10, strict_dir_limit=10)

    if len(output) < MAX_OUTPUT_CHARS:
        return output

    # 4. Critical Failure
    return (f"CRITICAL: Working directory is too large to represent in context. "
            f"Even with strict truncation, the tree exceeds {MAX_TREE_TOKENS} tokens. "
            "Please reorganize your files or use a .gitignore to exclude data directories.")


def get_runtime_info():
    """Gathers real-time system usage and directory structure."""
    
    # System usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    svmem = psutil.virtual_memory()
    mem_percent = svmem.percent
    
    usage_parts = [
        f"CPU Usage: {cpu_percent}%",
        f"Memory Usage: {mem_percent}% ({svmem.available / (1024**3):.2f} GB available)"
    ]
    
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, NVMLError
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        if gpu_count > 0:
            for i in range(gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                usage_parts.append(f"GPU {i} Utilization: {util.gpu}%")
                usage_parts.append(f"GPU {i} Memory: {(mem_info.used / mem_info.total) * 100:.1f}% used ({(mem_info.free / (1024**2)):.0f} MB free)")
        else:
            usage_parts.append("GPU: No NVIDIA GPUs detected.")
    except (ImportError, NVMLError):
        usage_parts.append("GPU: NVIDIA driver/pynvml not installed or failed to initialize.")
        
    system_usage = "\n".join(usage_parts)
    
    # Directory tree
    try:
        dir_tree = get_directory_tree_str('.')
    except RuntimeError as e:
        dir_tree = f"ERROR GENERATING DIRECTORY TREE: {e}"
    
    return f"""# Real-time System State
{system_usage}

# Project Directory Tree
{dir_tree}"""
