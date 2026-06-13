import sys
import os
from pathlib import Path

# Ensure the project root is in the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from aeon.tools.loader import load_tools_from_directory
    print("--- STARTUP OUTPUT SIMULATION ---")
    # We call this with verbose=True to trigger the print statements
    load_tools_from_directory(verbose=True)
    print("--- END SIMULATION ---")
except Exception as e:
    print(f"Error during verification: {e}")
    sys.exit(1)