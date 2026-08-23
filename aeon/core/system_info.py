import psutil
import json
import os
import socket
import subprocess
import time
import fnmatch
from datetime import datetime
from aeon.core.paths import RUNTIME_ROOT

STARTUP_DIR = RUNTIME_ROOT

# ===========================================================================
# Project Tree Configuration (ported from bc_llm_tools/read.py)
# ===========================================================================
HARD_IGNORE = {'.git', '.DS_Store', '.previous_objective.txt'}
SOFT_IGNORE = {
    '__pycache__', 'node_modules', 'venv', '.venv', '.ipynb_checkpoints',
    '.vscode', '.idea', 'dist', 'site-packages', '.tox', '.mypy_cache',
    '.pytest_cache', '.ruff_cache', 'egg-info',
    # Agent bookkeeping (sub-agent dirs, job logs, session state): accessed via
    # dedicated tools/digests, never navigated by tree. Without this it is only
    # excluded when the workspace's .gitignore happens to list it, so in a fresh
    # workspace it got walked into the prompt every iteration.
    'aeon_output',
}

BINARY_EXT = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.bmp', '.webp', '.tiff', '.svg',
    # Audio/Video
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv',
    # Compiled
    '.pyc', '.pyo', '.so', '.o', '.a', '.dylib', '.dll', '.exe',
    # ML/Data
    '.pt', '.pth', '.onnx', '.safetensors', '.bin', '.pkl', '.pickle',
    '.h5', '.hdf5', '.npy', '.npz', '.parquet', '.feather',
    # Archives
    '.zip', '.tar', '.gz', '.bz2', '.rar', '.7z', '.whl', '.tgz',
    # Documents
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    # Bioinformatics
    '.pdb', '.cif', '.sdf', '.mol2', '.bam', '.sam', '.vcf', '.bed',
    '.fast5', '.fasta', '.fastq', '.dcd', '.xtc', '.trr',
    # Fonts
    '.ttf', '.otf', '.woff', '.woff2',
    # Logs / build artifacts
    '.log', '.out', '.err',
}

PACKAGE_MANIFESTS = {
    'setup.py', 'pyproject.toml', 'setup.cfg', 'package.json',
    'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', 'CMakeLists.txt',
    'Makefile.PL', 'composer.json', 'Gemfile',
}

FILE_GROUP_THRESHOLD = 8   # Group files of same ext if more than this
MAX_DIR_ENTRIES = 50       # Max items shown per single directory
MAX_TREE_DEPTH = 8
MAX_TREE_CHARS = 50000     # Hard cap on total tree output


def _load_extra_ignores(startpath):
    """Load .gitignore patterns as additional soft ignores."""
    patterns = set()
    gitignore_path = os.path.join(startpath, '.gitignore')
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.add(line.rstrip('/'))
        except Exception:
            pass
    return patterns


def _is_soft_ignored(name, soft_ignore_set):
    """Check if name matches any soft ignore pattern (exact or fnmatch)."""
    if name in soft_ignore_set:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in soft_ignore_set)


def _is_external_package(dirpath):
    """Heuristic: is this directory an external/vendored package?

    Tier 1: has .git (cloned repo or submodule) -> auto-collapse.
    Tier 2: score >= 3 from LICENSE(+2) + manifest(+2) + README(+1) -> auto-collapse.
    """
    if os.path.exists(os.path.join(dirpath, '.git')):
        return True
    try:
        children = {e.name for e in os.scandir(dirpath)}
    except OSError:
        return False
    score = 0
    if any(n.upper().startswith('LICENSE') or n.upper().startswith('LICENCE') for n in children):
        score += 2
    if children & PACKAGE_MANIFESTS:
        score += 2
    if any(n.upper().startswith('README') for n in children):
        score += 1
    return score >= 3


def _format_size(size_bytes):
    """Format file size compactly."""
    if size_bytes < 1024:
        return f'{size_bytes}B'
    elif size_bytes < 1024 * 1024:
        return f'{size_bytes / 1024:.0f}K'
    else:
        return f'{size_bytes / (1024 * 1024):.1f}M'


def get_directory_tree_str(startpath='.', max_depth=None):
    """Build a compact, intelligent directory tree string.

    Features (ported from bc_llm_tools/read.py):
    - Groups files by extension when many of same type exist in a dir
    - Auto-collapses detected external packages (cloned repos, vendored deps)
    - Respects .gitignore patterns
    - Collapses binary/data file listings
    - Caps total output size to prevent context blowup
    """
    if max_depth is None:
        max_depth = MAX_TREE_DEPTH

    startpath = os.path.abspath(startpath)
    extra_ignores = _load_extra_ignores(startpath)
    all_soft = SOFT_IGNORE | extra_ignores

    lines = []
    char_count = [0]
    truncated = [False]

    def _add(line):
        if truncated[0]:
            return
        char_count[0] += len(line) + 1
        if char_count[0] > MAX_TREE_CHARS:
            lines.append('... [TREE TRUNCATED — too large] ...')
            truncated[0] = True
            return
        lines.append(line)

    def _walk(dirpath, depth=0):
        if truncated[0]:
            return
        if depth > max_depth:
            _add(f"{'    ' * depth}... [depth limit]")
            return

        try:
            entries = list(os.scandir(dirpath))
        except OSError:
            return

        # Separate dirs and files, skip hard ignores
        dirs = []
        files = []
        for e in entries:
            if e.name in HARD_IGNORE:
                continue
            try:
                if e.is_dir(follow_symlinks=False):
                    dirs.append(e)
                elif e.is_file(follow_symlinks=False):
                    files.append(e)
            except OSError:
                continue

        dirs.sort(key=lambda e: e.name)
        files.sort(key=lambda e: e.name)
        indent = '    ' * depth

        # --- Directories ---
        if len(dirs) > MAX_DIR_ENTRIES:
            omitted = len(dirs) - MAX_DIR_ENTRIES
            dirs = dirs[:MAX_DIR_ENTRIES]
            _add(f'{indent}... [{omitted} more directories omitted]')

        for d in dirs:
            if _is_soft_ignored(d.name, all_soft):
                _add(f'{indent}{d.name}/ (ignored)')
                continue

            # External package auto-collapse (non-root only)
            if depth > 0 and _is_external_package(d.path):
                try:
                    sub_count = sum(1 for _ in os.scandir(d.path))
                except OSError:
                    sub_count = '?'
                _add(f'{indent}{d.name}/ (external package, {sub_count} items)')
                continue

            _add(f'{indent}{d.name}/')
            _walk(d.path, depth + 1)

        # --- Files: group by extension when many of same type ---
        ext_groups = {}
        for f in files:
            ext = os.path.splitext(f.name)[1].lower()
            if ext not in ext_groups:
                ext_groups[ext] = []
            ext_groups[ext].append(f)

        IMPORTANT_NAMES = {
            'README.md', 'README.txt', 'Dockerfile', 'Makefile',
            'requirements.txt', 'setup.py', 'package.json', '__init__.py',
            'main.py', 'pyproject.toml', '.gitignore', '.dockerignore',
        }

        individual_files = []
        group_lines = []

        for ext, flist in sorted(ext_groups.items()):
            if len(flist) > FILE_GROUP_THRESHOLD:
                # Show important files individually, group the rest
                important = [f for f in flist if f.name in IMPORTANT_NAMES]
                remainder = len(flist) - len(important)
                individual_files.extend(important)
                if remainder > 0:
                    try:
                        total_sz = sum(f.stat().st_size for f in flist if f not in important)
                    except OSError:
                        total_sz = 0
                    group_lines.append(
                        f'{indent}[{remainder} {ext or "no-ext"} files, {_format_size(total_sz)} total]'
                    )
            else:
                individual_files.extend(flist)

        # Cap individual file listings
        if len(individual_files) > MAX_DIR_ENTRIES:
            individual_files = individual_files[:MAX_DIR_ENTRIES]
            _add(f'{indent}... [more files omitted]')

        for f in sorted(individual_files, key=lambda e: e.name):
            try:
                sz = f.stat().st_size
                ext = os.path.splitext(f.name)[1].lower()
                sz_str = _format_size(sz)
                if ext in BINARY_EXT:
                    _add(f'{indent}{f.name} ({sz_str}, binary)')
                else:
                    _add(f'{indent}{f.name} ({sz_str})')
            except OSError:
                _add(f'{indent}{f.name} (?)')

        for gl in group_lines:
            _add(gl)

    _walk(startpath, depth=0)

    if not lines:
        lines.append('(empty directory)')

    return '\n'.join(lines)


_COORDINATOR = "/home/aday/website_hosting/gpu_coord.py"
_COORDINATOR_CWD = "/home/aday/website_hosting/ads"
_COORDINATOR_HOSTNAME = "DAY2RTX6000PRO"
_COORDINATOR_HOST = "192.168.0.177"
_GPU_STATUS_CACHE = (0.0, [])


def _coordinator_gpu_parts():
    """Return sanitized local GPU status from the sole fleet control plane.

    Direct NVML discovery is forbidden on renter hosts because ACL-hidden devices
    can be renumbered or disappear.  Physical numbers below are explicitly
    diagnostic labels from coordinator output, never CUDA selectors.
    """
    global _GPU_STATUS_CACHE
    if socket.gethostname() != _COORDINATOR_HOSTNAME:
        return ["gpu: coordinator view available only on .177"]
    now = time.monotonic()
    cached_at, cached = _GPU_STATUS_CACHE
    if cached and now - cached_at < 10:
        return list(cached)
    try:
        result = subprocess.run(
            ["python3", _COORDINATOR, "status", "--json"],
            cwd=_COORDINATOR_CWD,
            capture_output=True,
            text=True,
            timeout=12,
            check=True,
        )
        inventory = json.loads(result.stdout)
        if not isinstance(inventory, list):
            raise ValueError("coordinator status is not a list")
        parts = []
        for item in inventory:
            if not isinstance(item, dict) or item.get("host") != _COORDINATOR_HOST:
                continue
            try:
                physical = int(item["physical_gpu"])
            except (KeyError, TypeError, ValueError):
                continue
            state = str(item.get("state") or "UNKNOWN").upper()
            if not all(ch.isalnum() or ch == "_" for ch in state):
                state = "UNKNOWN"
            label = f"physical-gpu{physical}(diagnostic): {state}"
            util = item.get("utilization_pct")
            capacity = item.get("vram_share_capacity_mib")
            if state in {"AVAILABLE", "SHARED_AVAILABLE", "RESERVED", "RESERVED_RUNNING"}:
                try:
                    label += f" {max(0, min(100, float(util))):.0f}% util"
                except (TypeError, ValueError):
                    pass
                try:
                    label += f" {max(0.0, float(capacity)) / 1024.0:.1f}gb allocatable"
                except (TypeError, ValueError):
                    pass
            parts.append(label)
        if not parts:
            parts = ["gpu: no local coordinator inventory"]
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError):
        return ["gpu: coordinator status unavailable"]
    _GPU_STATUS_CACHE = (now, list(parts))
    return parts


def get_system_stats() -> str:
    """The small, VOLATILE per-turn stats line (datetime, cpu, mem, gpu). Kept
    separate from the project tree so the prompt can place the (semi-static) tree
    in its cacheable prefix while these few churning lines live in the volatile
    tail — otherwise a per-turn timestamp busts the tree's prefix cache."""
    # Minute precision (not seconds): the model never needs sub-minute resolution,
    # and a per-second value is pure churn.
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    # interval=None is non-blocking and reports utilization since the previous
    # call (~one iteration ago) — both faster (no 100ms/turn stall) and a more
    # meaningful per-iteration figure than a 0.1s sample.
    cpu_percent = psutil.cpu_percent(interval=None)
    svmem = psutil.virtual_memory()
    parts = [
        f'datetime: {now_str}',
        f'cpu: {cpu_percent}%',
        f'mem: {svmem.percent}% ({svmem.available / (1024 ** 3):.1f}gb free)',
    ]
    parts.extend(_coordinator_gpu_parts())
    return f"**STATS**\n{' | '.join(parts)}"


def get_project_tree() -> str:
    """The SEMI-STATIC workspace path + project tree. Changes only when files
    change (not every turn), so it belongs in the cacheable prompt prefix rather
    than being regenerated and re-prefilled every step."""
    dir_tree = get_directory_tree_str('.')
    return f"**WORKSPACE**\n{STARTUP_DIR}\n\n**PROJECT TREE**\n{dir_tree}"


def get_runtime_info() -> str:
    """Backward-compatible combined view (stats + tree)."""
    return f"{get_system_stats()}\n\n{get_project_tree()}"
