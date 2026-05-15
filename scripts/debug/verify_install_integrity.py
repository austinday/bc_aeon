import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        print(f"Error: {res.stderr}")
    return res

def main():
    project_root = "/home/aday/bc_aeon"
    python_exe = sys.executable
    
    print(f"--- Installation Integrity Check ---")
    print(f"Python Executable: {python_exe}")
    
    # 1. Clean uninstall
    print("\n[1/4] Uninstalling aeon...")
    run_cmd([python_exe, "-m", "pip", "uninstall", "-y", "aeon"])
    
    # 2. Clean install
    print("\n[2/4] Installing aeon from project root...")
    install_res = run_cmd([python_exe, "-m", "pip", "install", "."], cwd=project_root)
    if install_res.returncode != 0:
        print("Installation failed!")
        sys.exit(1)
    print("Installation successful.")
    
    # 3. Verify installation location
    print("\n[3/4] Checking pip show...")
    show_res = run_cmd([python_exe, "-m", "pip", "show", "aeon"])
    print(show_res.stdout)
    
    # 4. Attempt import from /tmp
    print("\n[4/4] Attempting import from /tmp...")
    # We create a small script in /tmp to run
    test_script_path = "/tmp/aeon_import_test.py"
    with open(test_script_path, "w") as f:
        f.write(f"""
import sys
import os
print(f"CWD: {{os.getcwd()}}")
print(f"sys.path: {{sys.path}}")
try:
    import aeon
    print(f"SUCCESS: aeon imported from {{aeon.__file__}}")
    from aeon.core.worker import Worker
    print("SUCCESS: aeon.core.worker imported")
except Exception as e:
    print(f"FAILURE: {{e}}")
    sys.exit(1)
""")
    
    import_res = run_cmd([python_exe, test_script_path], cwd="/tmp")
    print(import_res.stdout)
    print(import_res.stderr)
    
    if import_res.returncode != 0:
        print("\nRESULT: Installation is BROKEN.")
        sys.exit(1)
    else:
        print("\nRESULT: Installation is VALID.")

if __name__ == "__main__":
    main()