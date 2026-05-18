import subprocess
import sys
import os
import shutil
from pathlib import Path

def log(msg):
    print(f"[TEST] {msg}")

def main():
    project_root = "/home/aday/bc_aeon"
    python_exe = sys.executable
    
    log(f"Using Python: {python_exe}")
    log(f"Project Root: {project_root}")

    # 1. Uninstall existing aeon
    log("Uninstalling aeon...")
    subprocess.run([python_exe, '-m', 'pip', 'uninstall', '-y', 'aeon'], capture_output=True)

    # 2. Install aeon using the exact method in _execute_restart
    log("Installing aeon via 'pip install . '...")
    try:
        # We must be in the project root for 'pip install .' to work
        os.chdir(project_root)
        result = subprocess.run(
            [python_exe, '-m', 'pip', 'install', '.', '--upgrade', '--force-reinstall', '--no-cache-dir'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            log(f"Installation failed with code {result.returncode}")
            log(f"Stdout: {result.stdout}")
            log(f"Stderr: {result.stderr}")
            sys.exit(1)
        log("Installation command succeeded.")
    except Exception as e:
        log(f"Installation exception: {e}")
        sys.exit(1)

    # 3. Verify installation from a neutral directory (/tmp)
    log("Verifying installation from /tmp...")
    os.chdir('/tmp')
    
    # We use a separate process to ensure we aren't using cached imports in the current process
    verify_script = """
import sys
import os
try:
    import aeon
    print(f'SUCCESS: aeon found at {aeon.__file__}')
    from aeon.core import logger
    print(f'SUCCESS: aeon.core.logger found at {logger.__file__}')
    from aeon.core.worker import Worker
    print('SUCCESS: aeon.core.worker.Worker imported')
except ImportError as e:
    print(f'FAILURE: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"""
    result = subprocess.run(
        [python_exe, '-c', verify_script],
        capture_output=True,
        text=True
    )
    
    print("\n--- Verification Result ---")
    print(result.stdout)
    print(result.stderr)
    
    if result.returncode == 0:
        log("Verification PASSED.")
    else:
        log("Verification FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()