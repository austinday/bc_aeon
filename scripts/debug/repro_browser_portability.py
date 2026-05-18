import os
import sys
import subprocess

# Add project root to sys.path to allow importing aeon
project_root = "/home/aday/bc_aeon"
sys.path.append(project_root)

try:
    from aeon.tools.browser import ensure_browser_running
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def reproduce():
    print(f"Initial CWD: {os.getcwd()}")
    
    # Change directory to something outside the project root
    os.chdir('/tmp')
    print(f"Changed CWD to: {os.getcwd()}")
    
    print("Attempting to ensure browser is running...")
    try:
        ensure_browser_running()
        print("Successfully started browser!")
    except subprocess.CalledProcessError as e:
        print(f"Caught Expected Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Caught Unexpected Error: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    reproduce()