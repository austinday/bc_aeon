import psutil
import os
import time

MAX_TREE_TOKENS = 100000 
CHARS_PER_TOKEN = 4
MAX_OUTPUT_CHARS = MAX_TREE_TOKENS * CHARS_PER_TOKEN

def get_directory_tree_str(startpath='.', strict_file_limit=None, strict_dir_limit=None):
    all_files_stats = []
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        files = [f for f in files if not f.startswith('.')]
        for f in files:
            fp = os.path.join(root, f)
            try:
                st = os.stat(fp)
                all_files_stats.append({'path': fp, 'size': st.st_size, 'mtime': st.st_mtime})
            except OSError as e:
                # Log but continue - file may have been deleted during walk
                pass
            
    all_files_stats.sort(key=lambda x: x['mtime'], reverse=True)
    top_10_recent_paths = set(x['path'] for x in all_files_stats[:10])
    
    now = time.time()
    tree_lines = []
    tree_lines.append(f"ref_time: {int(now)} | units: MB | timestamps: top 10 most recent (seconds from ref)")
    
    for root, dirs, files in os.walk(startpath, topdown=True):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        files = [f for f in files if not f.startswith('.')]
        dirs.sort(); files.sort()
        if strict_dir_limit is not None and len(dirs) > strict_dir_limit: 
            dirs[:] = dirs[:strict_dir_limit]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        if root != '.': 
            tree_lines.append(f'{indent}{os.path.basename(root)}/')
            
        sub_indent = ' ' * 4 * (level + 1)
        files_to_display = files
        if strict_file_limit is not None and len(files) > strict_file_limit:
            files_to_display = files[:strict_file_limit]
            tree_lines.append(f"{sub_indent}... [{len(files) - strict_file_limit} omitted]")
            
        for f in files_to_display:
            full_path = os.path.join(root, f)
            try:
                stat = os.stat(full_path)
                size_mb = stat.st_size / (1024 * 1024)
                # Use compact format: show decimal only if < 10MB
                if size_mb < 0.01:
                    meta = "<0.01"
                elif size_mb < 10:
                    meta = f"{size_mb:.2f}"
                else:
                    meta = f"{size_mb:.1f}"
                if full_path in top_10_recent_paths:
                    meta += f", {int(stat.st_mtime - now)}"
                tree_lines.append(f'{sub_indent}{f} ({meta})')
            except OSError:
                tree_lines.append(f'{sub_indent}{f} (?)')
    return "\n".join(tree_lines)

def get_runtime_info():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    svmem = psutil.virtual_memory()
    parts = [f"cpu: {cpu_percent}%", f"mem: {svmem.percent}% ({svmem.available/(1024**3):.1f}gb free)"]
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, NVMLError
        nvmlInit()
        for i in range(nvmlDeviceGetCount()):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                parts.append(f"gpu{i}: {util.gpu}% ({mem.free/(1024**3):.1f}gb free)")
            except Exception:
                parts.append(f"gpu{i}: n/a")
    except ImportError:
        parts.append("gpu: n/a (pynvml not installed)")
    except Exception:
        parts.append("gpu: n/a")
    dir_tree = get_directory_tree_str('.')
    return f"**stats:** {' | '.join(parts)}\n\n**project tree**\n{dir_tree}"
