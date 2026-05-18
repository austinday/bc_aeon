import sys
import os
import subprocess
from pathlib import Path

def diag():
    print("=== RESTART DIAGNOSTIC ===")
    print(f"PID: {os.getpid()}")
    print(f"CWD: {os.getcwd()}")
    print(f"Python Executable: {sys.executable}")
    print(f"sys.path: {sys.path}")
    
    # Check for PYTHONPATH
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
    
    try:
        import aeon
        print(f"SUCCESS: Imported aeon")
        print(f"aeon.__file__: {aeon.__file__}")
        
        # Check if it's the source or the installed package
        if 'site-packages' in aeon.__file__:
            print("aeon is being imported from site-packages")
        else:
            print("aeon is being imported from source/local directory")
            
        try:
            from aeon.core import logger
            print("SUCCESS: Imported aeon.core.logger")
        except ImportError as e:
            print(f"FAIL: Could not import aeon.core.logger: {e}")
            # List contents of the aeon package directory
            aeon_dir = Path(aeon.__file__).parent
            print(f"Contents of {aeon_dir}:")
            for item in aeon_dir.iterdir():
                print(f"  - {item.name}")
                
    except ImportError as e:
        print(f"FAIL: Could not import aeon: {e}")

    print("==========================")

if __name__ == "__main__":
    diag()