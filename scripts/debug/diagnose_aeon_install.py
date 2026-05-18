import sys
import os
import subprocess
import shutil
from pathlib import Path

def print_sep(title):
    print(f"\n{'='*20} {title} {'='*20}")

def check_aeon_location():
    print_sep("Checking Aeon Location")
    try:
        import aeon
        print(f"Aeon module path: {aeon.__file__}")
        print(f"Aeon version: {getattr(aeon, '__version__', 'Unknown')}")
    except ImportError:
        print("Aeon NOT installed in current environment.")

def run_smoke_test_from_tmp():
    print_sep("Running Smoke Test from /tmp")
    smoke_test_path = os.path.abspath("aeon/smoke_test.py")
    if not os.path.exists(smoke_test_path):
        print(f"Smoke test not found at {smoke_test_path}")
        return

    try:
        # Run as a subprocess from /tmp to avoid local directory imports
        result = subprocess.run(
            [sys.executable, "-B", smoke_test_path],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"Return Code: {result.returncode}")
        print(f"Stdout:\n{result.stdout}")
        print(f"Stderr:\n{result.stderr}")
    except Exception as e:
        print(f"Error running smoke test: {e}")

def simulate_restart_install():
    print_sep("Simulating Restart Installation (pip install .)")
    
    # 1. Uninstall
    print("Uninstalling aeon...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "aeon"], capture_output=True)
    
    # 2. Install
    print("Installing aeon via 'pip install . '...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "."],
        cwd=os.getcwd(),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Install failed:\n{result.stderr}")
        return

    print("Install successful. Now checking location and running smoke test...")
    check_aeon_location()
    run_smoke_test_from_tmp()

if __name__ == "__main__":
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    print_sep("Initial State")
    check_aeon_location()
    run_smoke_test_from_tmp()
    
    simulate_restart_install()