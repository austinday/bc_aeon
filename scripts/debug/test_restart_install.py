import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)} in {cwd or os.getcwd()}")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return res

def test_install(mode="editable"):
    print(f"\n--- Testing {'Editable' if mode == 'editable' else 'Regular'} Install ---")
    
    # 1. Uninstall existing
    run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "aeon"])
    
    # 2. Install
    install_cmd = [sys.executable, "-m", "pip", "install"]
    if mode == "editable":
        install_cmd.extend(["-e", "."])
    else:
        install_cmd.append(".")
    
    res = run_cmd(install_cmd, cwd="/home/aday/bc_aeon")
    if res.returncode != 0:
        print(f"Install failed: {res.stderr}")
        return False

    # 3. Verify import from neutral directory
    # We use a separate process to avoid cached imports
    verify_cmd = [sys.executable, "-c", "import aeon.core.logger; print('Import Success')"]
    # Run from /tmp to ensure we aren't just picking up the local directory
    res = run_cmd(verify_cmd, cwd="/tmp")
    
    if res.returncode == 0:
        print("Import Success!")
        return True
    else:
        print(f"Import Failed: {res.stderr}")
        return False

if __name__ == "__main__":
    # Ensure we are in the project root for the install
    os.chdir("/home/aday/bc_aeon")
    
    # Test editable
    editable_ok = test_install("editable")
    
    # Test regular
    regular_ok = test_install("regular")
    
    print("\n--- RESULTS ---")
    print(f"Editable Install: {'PASS' if editable_ok else 'FAIL'}")
    print(f"Regular Install:  {'PASS' if regular_ok else 'FAIL'}")
    
    if not editable_ok and regular_ok:
        print("\nHYPOTHESIS CONFIRMED: Editable install is failing where regular install succeeds.")
        sys.exit(1)
    elif editable_ok and regular_ok:
        print("\nBoth succeeded. The issue might be more subtle (e.g., environment mismatch during execv).")
        sys.exit(0)
    else:
        print("\nBoth failed or regular failed. Something else is wrong.")
        sys.exit(1)