import subprocess
import sys
import os
import shutil

def test_install(use_user=False):
    print(f"Testing installation with use_user={use_user}...")
    
    # 1. Clean up
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'aeon'], capture_output=True)
    
    # 2. Install
    cmd = [sys.executable, '-m', 'pip', 'install']
    if use_user:
        cmd.append('--user')
    cmd.append('.')
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd='/home/aday/bc_aeon', capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Install failed: {result.stderr}")
        return False

    # 3. Verify import from /tmp
    print("Verifying import from /tmp...")
    verify_cmd = [sys.executable, '-B', '-c', 'import aeon; print(aeon.__file__)']
    verify_result = subprocess.run(verify_cmd, cwd='/tmp', capture_output=True, text=True)
    
    if verify_result.returncode == 0:
        print(f"SUCCESS: Imported aeon from {verify_result.stdout.strip()}")
        return True
    else:
        print(f"FAILED: Could not import aeon. Error: {verify_result.stderr}")
        return False

if __name__ == "__main__":
    # Test regular install
    res_reg = test_install(use_user=False)
    print(f"Regular install result: {'PASS' if res_reg else 'FAIL'}\n")
    
    # Test user install
    res_user = test_install(use_user=True)
    print(f"User install result: {'PASS' if res_user else 'FAIL'}")