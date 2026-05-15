import subprocess
import os
import sys
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result

def main():
    project_root = "/home/aday/bc_aeon"
    tmp_dir = "/tmp"
    
    print("--- Starting Full Restart Sequence Reproduction ---")
    print(f"Project Root: {project_root}")
    
    # 1. Uninstall aeon
    print("\n[1] Uninstalling aeon...")
    run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "aeon"])
    
    # 2. Install aeon from project root
    print("\n[2] Installing aeon from project root...")
    install_res = run_cmd([sys.executable, "-m", "pip", "install", ".", "--upgrade", "--force-reinstall", "--no-cache-dir"], cwd=project_root)
    if install_res.returncode != 0:
        print(f"Install failed: {install_res.stderr}")
        return

    # 3. Attempt to import from /tmp (simulating the relaunch environment)
    print("\n[3] Attempting to import aeon.core.logger from /tmp...")
    
    # We use a separate process to ensure a clean import state
    import_code = """
import sys
import os
print(f'CWD: {os.getcwd()}')
print(f'sys.path: {sys.path}')
try:
    import aeon
    print(f'aeon found at: {aeon.__file__}')
    from aeon.core import logger
    print('SUCCESS: Imported aeon.core.logger')
except ImportError as e:
    print(f'FAILURE: {e}')
    sys.exit(1)
except Exception as e:
    print(f'Unexpected error: {e}')
    sys.exit(1)
"""
    import_res = subprocess.run(
        [sys.executable, "-B", "-c", import_code],
        capture_output=True, text=True, cwd=tmp_dir
    )
    
    print(f"STDOUT:\n{import_res.stdout}")
    print(f"STDERR:\n{import_res.stderr}")
    
    if import_res.returncode == 0:
        print("\nRESULT: Could NOT reproduce the error. The package is importable from /tmp.")
    else:
        print("\nRESULT: REPRODUCED the error! The package is NOT importable from /tmp.")

if __name__ == "__main__":
    main()