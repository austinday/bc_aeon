import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    print(f"Return Code: {result.returncode}")
    if result.stdout: print(f"STDOUT:\n{result.stdout}")
    if result.stderr: print(f"STDERR:\n{result.stderr}")
    return result

def main():
    project_root = "/home/aday/bc_aeon"
    python_exe = sys.executable
    
    print("--- Step 1: Uninstalling existing aeon ---")
    run_cmd([python_exe, "-m", "pip", "uninstall", "-y", "aeon"])
    
    print("\n--- Step 2: Installing aeon via 'pip install .' ---")
    install_res = run_cmd([python_exe, "-m", "pip", "install", ".", "--upgrade", "--force-reinstall", "--quiet"], cwd=project_root)
    
    if install_res.returncode != 0:
        print("CRITICAL: Installation failed!")
        return

    print("\n--- Step 3: Verifying import from /tmp ---")
    # We use a separate process to ensure we aren't using the current process's sys.path
    verify_script = "/tmp/verify_aeon_import.py"
    with open(verify_script, "w") as f:
        f.write(f"""
import sys
import os
print(f"Python Executable: {{sys.executable}}")
print(f"CWD: {{os.getcwd()}}")
print(f"sys.path: {{sys.path}}")
try:
    import aeon
    print(f"SUCCESS: Imported aeon from {{aeon.__file__}}")
    from aeon.core import logger
    print("SUCCESS: Imported aeon.core.logger")
    from aeon.core import worker
    print("SUCCESS: Imported aeon.core.worker")
except ImportError as e:
    print(f"FAILURE: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
""")
    
    verify_res = run_cmd([python_exe, "-B", verify_script], cwd="/tmp")
    
    if verify_res.returncode == 0:
        print("\nRESULT: Installation is VALID and importable from /tmp.")
    else:
        print("\nRESULT: Installation is INVALID or not importable from /tmp.")

if __name__ == "__main__":
    main()