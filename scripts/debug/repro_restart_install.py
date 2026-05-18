import subprocess
import sys
import os
import shutil
from pathlib import Path

def log(msg):
    print(f"[REPRO] {msg}")

def run_cmd(cmd, cwd=None, env=None):
    log(f"Running: {' '.join(cmd)} (cwd={cwd})")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if res.returncode != 0:
        log(f"FAILED: {res.stderr}")
    else:
        log(f"SUCCESS")
    return res

def main():
    # 1. Setup paths
    project_root = "/home/aday/bc_aeon"
    python_exe = sys.executable
    
    log(f"Starting reproduction. Python: {python_exe}")
    
    # 2. Uninstall existing aeon
    log("Step 1: Uninstalling aeon...")
    run_cmd([python_exe, "-m", "pip", "uninstall", "-y", "aeon"])
    
    # 3. Install aeon from root
    log("Step 2: Installing aeon from root...")
    run_cmd([python_exe, "-m", "pip", "install", "."], cwd=project_root)
    
    # 4. Verify installation with pip show
    log("Step 3: Verifying installation with pip show...")
    show = run_cmd([python_exe, "-m", "pip", "show", "aeon"])
    print(show.stdout)
    
    # 5. Run smoke test from /tmp with clean environment
    log("Step 4: Running smoke test from /tmp...")
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    
    # We use -m aeon.smoke_test. 
    # To ensure we aren't just picking up the local folder, we MUST be in /tmp
    # and the local folder MUST NOT be in sys.path.
    
    smoke = run_cmd(
        [python_exe, "-B", "-m", "aeon.smoke_test"],
        cwd="/tmp",
        env=clean_env
    )
    
    print("--- SMOKE TEST OUTPUT ---")
    print(smoke.stdout)
    print(smoke.stderr)
    print("------------------------")
    
    if smoke.returncode == 0:
        log("REPRODUCTION RESULT: Smoke test PASSED")
    else:
        log("REPRODUCTION RESULT: Smoke test FAILED")

if __name__ == "__main__":
    main()