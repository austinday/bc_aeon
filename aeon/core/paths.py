import os
from pathlib import Path
from typing import Optional

def resolve_project_root() -> Path:
    """
    Resolves the root directory of the aeon project.
    
    Logic:
    1. Check AEON_PROJECT_ROOT environment variable.
    2. Look for the directory containing 'setup.py' by walking up from this file.
    3. Fallback to the current working directory.
    """
    # 1. Environment Variable Override
    env_root = os.environ.get("AEON_PROJECT_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists() and (p / "setup.py").exists():
            return p

    # 2. Walk up from this file (aeon/core/paths.py)
    # This file is at <root>/aeon/core/paths.py
    # Root is 2 levels up from 'core', 3 from 'paths.py'
    current_file = Path(__file__).resolve()
    
    # Search upwards for setup.py to find the actual project root
    for parent in [current_file] + list(current_file.parents):
        if (parent / "setup.py").exists():
            return parent
            
    # 3. Final Fallback: CWD
    return Path(os.getcwd()).resolve()

# Global constants for the session
PROJECT_ROOT = resolve_project_root()
# Runtime root is where the user is executing from, or a designated output dir
RUNTIME_ROOT = Path(os.getcwd()).resolve()

def get_project_root() -> Path:
    return PROJECT_ROOT

def get_runtime_root() -> Path:
    return RUNTIME_ROOT

def get_output_dir() -> Path:
    """Returns the directory for logs and agent outputs, relative to runtime root."""
    return RUNTIME_ROOT / "aeon_output"