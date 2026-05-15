import subprocess
import sys
import os
import shutil
from pathlib import Path

def run(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)} (cwd={cwd})")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        print(f"FAILED:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
    else:
        print("SUCCESS")
    return res

def main():
    project_root = "/home/aday/bc_aeon"
    python_exe = sys.executable
    
    print("--- Step 1: Explicit Uninstall ---")
    run([python_exe, "-m", "pip", "uninstall", "-y", "aeon"])
    
    print("\n--- Step 2: Regular Install ---")
    run([python_exe, "-m", "pip", "install", "."], cwd=project_root)
    
    print("\n--- Step 3: Verify Import from /tmp ---")
    # We use a separate process to ensure no CWD leakage
    verify_cmd = [
        python_exe, "-c", 
        "import sys; import os; os.chdir('/tmp'); import aeon.core.logger; print('Import successful!')"
    ]
    res = run(verify_cmd)
    
    if res.returncode == 0:
        print("\nVERIFICATION PASSED: Package is importable from /tmp")
    else:
        print("\nVERIFICATION FAILED: Package still not importable")
        sys.exit(1)

if __name__ == "__main__":
    main()