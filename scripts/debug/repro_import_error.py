import subprocess
import sys
import os
from pathlib import Path

def run_test():
    project_root = "/home/aday/bc_aeon"
    
    print(f"--- Starting Reproduction Test ---")
    print(f"Project Root: {project_root}")
    
    # 1. Uninstall aeon to simulate the state after 'pip uninstall' in _execute_restart
    print("\n[1] Uninstalling aeon...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'aeon'], capture_output=True)
    
    # 2. Attempt to import aeon.core.logger from the project root
    print("\n[2] Attempting to import aeon.core.logger from project root...")
    
    # We use a subprocess to ensure a clean import environment
    # We mimic 'python3 -m aeon.main' by setting CWD to project root and running a script
    test_code = """
import sys
import os
# Mimic the CWD being the project root
sys.path.insert(0, os.getcwd())
try:
    import aeon
    print(f'SUCCESS: Imported aeon from {aeon.__file__}')
    from aeon.core import logger
    print(f'SUCCESS: Imported aeon.core.logger from {logger.__file__}')
except ImportError as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"""
    
    result = subprocess.run(
        [sys.executable, '-c', test_code],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    
    if result.returncode == 0:
        print("\nRESULT: Could NOT reproduce the error. Local import worked without pip install.")
    else:
        print("\nRESULT: Reproduced the error! Local import failed without pip install.")

if __name__ == "__main__":
    run_test()