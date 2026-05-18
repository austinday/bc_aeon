import subprocess
import os
import sys
from pathlib import Path

def main():
    print("=== REPRODUCING SMOKE TEST FAILURE ===")
    
    # Mimic the environment in _execute_restart
    clean_env = os.environ.copy()
    clean_env.pop('PYTHONPATH', None)
    
    # The command used in main.py
    cmd = [sys.executable, '-B', '-E', '-m', 'aeon.smoke_test']
    cwd = '/tmp'
    
    print(f"Executing: {' '.join(cmd)}")
    print(f"CWD: {cwd}")
    print(f"PYTHONPATH in env: {clean_env.get('PYTHONPATH', 'Not Set')}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
            env=clean_env
        )
        
        print("\n--- STDOUT ---")
        print(result.stdout)
        print("\n--- STDERR ---")
        print(result.stderr)
        print(f"\nReturn Code: {result.returncode}")
        
    except Exception as e:
        print(f"Error running command: {e}")

if __name__ == "__main__":
    main()