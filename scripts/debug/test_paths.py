import sys
import os
from pathlib import Path

# Add the project root to sys.path so we can import aeon
# This simulates the package being installed or the project being in the python path
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(current_dir))

try:
    from aeon.core.paths import PROJECT_ROOT, RUNTIME_ROOT, get_project_root
    
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Runtime Root: {RUNTIME_ROOT}")
    print(f"Getter Root: {get_project_root()}")
    
    # Verification logic
    expected_root = PROJECT_ROOT
    if "setup.py" in (expected_root / "setup.py").name if (expected_root / "setup.py").exists() else False:
        print("SUCCESS: Project root correctly identified by presence of setup.py")
    else:
        print("FAILURE: Project root does not contain setup.py")

except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)