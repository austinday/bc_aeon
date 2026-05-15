import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"[CMD] {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return res

def test_import(cwd=None):
    # We use a separate process to avoid polluting the current process's sys.modules
    cmd = [sys.executable, "-c", "import aeon; print(f'aeon: {aeon.__file__}'); from aeon.core import logger; print(f'logger: {logger.__file__}')"]
    res = run_cmd(cmd, cwd=cwd)
    print(res.stdout)
    print(res.stderr)
    return res.returncode == 0

def main():
    project_root = os.getcwd()
    print(f"Project Root: {project_root}")
    
    # 1. Clean state: Uninstall aeon
    print("\n--- Step 1: Uninstalling aeon ---")
    run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "aeon"])
    
    # Verify it's gone
    if test_import(cwd="/tmp"):
        print("Error: aeon still importable after uninstall")
    else:
        print("Success: aeon uninstalled")

    # 2. Install aeon
    print("\n--- Step 2: Installing aeon via 'pip install .' ---")
    install_res = run_cmd([sys.executable, "-m", "pip", "install", "."], cwd=project_root)
    if install_res.returncode != 0:
        print(f"Installation failed: {install_res.stderr}")
        sys.exit(1)
    print("Installation command succeeded")

    # 3. Test import from /tmp (Neutral directory)
    print("\n--- Step 3: Testing import from /tmp ---")
    if test_import(cwd="/tmp"):
        print("SUCCESS: Import worked from /tmp")
    else:
        print("FAIL: Import failed from /tmp")

    # 4. Test import from project root (CWD shadowing)
    print("\n--- Step 4: Testing import from project root ---")
    if test_import(cwd=project_root):
        print("SUCCESS: Import worked from project root")
    else:
        print("FAIL: Import failed from project root")

    # 5. Test running as a module from project root
    print("\n--- Step 5: Testing 'python3 -m aeon.main' from project root ---")
    # We just check if the module can be loaded
    cmd = [sys.executable, "-m", "aeon.main", "--help"] # Use --help to avoid starting the agent
    res = run_cmd(cmd, cwd=project_root)
    if res.returncode == 0 or "usage:" in res.stdout.lower():
        print("SUCCESS: Module loaded and executed")
    else:
        print(f"FAIL: Module failed to load: {res.stderr}")

if __name__ == "__main__":
    main()