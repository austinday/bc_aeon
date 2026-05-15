import subprocess
import sys
import os
import shutil
from pathlib import Path

def log(msg):
    print(f"[TRACE] {msg}")

def run_cmd(cmd, cwd=None):
    log(f"Running: {' '.join(cmd)} in {cwd or os.getcwd()}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        log(f"FAILED: {res.stderr}")
    return res

def main():
    project_root = "/home/aday/bc_aeon"
    
    # 1. Clean start: Uninstall any existing aeon
    log("Step 1: Uninstalling existing aeon...")
    run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "aeon"])
    
    # 2. Install from source
    log("Step 2: Installing aeon from source...")
    install_res = run_cmd([sys.executable, "-m", "pip", "install", "."], cwd=project_root)
    log(f"Install Output: {install_res.stdout}")
    
    # 3. Check pip show
    log("Step 3: Checking pip show aeon...")
    show_res = run_cmd([sys.executable, "-m", "pip", "show", "aeon"])
    log(f"Pip Show:\n{show_res.stdout}")
    
    # Extract location
    location = None
    for line in show_res.stdout.splitlines():
        if line.startswith("Location: "):
            location = line.split("Location: ")[1].strip()
    
    if not location:
        log("ERROR: Could not find installation location via pip show.")
        sys.exit(1)
    
    log(f"Installation location: {location}")
    
    # 4. Inspect installation directory
    log(f"Step 4: Inspecting contents of {location}...")
    try:
        # Look for the aeon folder in site-packages
        aeon_site_dir = Path(location) / "aeon"
        if aeon_site_dir.exists():
            log(f"Found aeon directory at {aeon_site_dir}")
            for item in aeon_site_dir.iterdir():
                log(f"  - {item.name}")
            
            core_dir = aeon_site_dir / "core"
            if core_dir.exists():
                log(f"Found core directory at {core_dir}")
                for item in core_dir.iterdir():
                    log(f"    - {item.name}")
            else:
                log("ERROR: 'core' directory missing from installed package!")
        else:
            log(f"ERROR: 'aeon' directory NOT found in {location}!")
    except Exception as e:
        log(f"Error inspecting directory: {e}")

    # 5. Attempt import from /tmp
    log("Step 5: Attempting import from /tmp...")
    # We use a separate process to ensure no CWD pollution
    import_cmd = [
        sys.executable, 
        "-B", 
        "-c", 
        "import aeon; print(f'Imported aeon from: {aeon.__file__}'); from aeon.core.worker import Worker; print('Successfully imported aeon.core.worker')"
    ]
    import_res = subprocess.run(import_cmd, capture_output=True, text=True, cwd="/tmp")
    
    if import_res.returncode == 0:
        log("SUCCESS: Import worked from /tmp!")
        log(f"Output: {import_res.stdout}")
    else:
        log("FAILED: Import failed from /tmp!")
        log(f"Error: {import_res.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()