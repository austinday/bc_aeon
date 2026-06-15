import os
from pathlib import Path
import sys

# Mocking the environment to match main.py's logic
def test_discovery():
    print("--- Skills Discovery Diagnostic ---")
    
    # Try to find the project root
    # In main.py, it's essentially the cwd or the dir containing 'aeon'
    cwd = Path(os.getcwd())
    print(f"CWD: {cwd}")
    
    # Simulate the path resolution in main.py/SkillsManager
    # main.py uses: sm = SkillsManager() -> sm.base_dir
    # SkillsManager uses: PROJECT_ROOT / "aeon" / "core" / "skills"
    
    # Let's try to find the 'aeon' package directory
    aeon_dir = cwd / "aeon"
    if not aeon_dir.exists():
        print("ERROR: 'aeon' directory not found in CWD")
        return

    skills_dir = aeon_dir / "core" / "skills"
    print(f"Target Skills Dir: {skills_dir}")
    
    if not skills_dir.exists():
        print(f"ERROR: Skills directory does not exist at {skills_dir}")
        return

    print(f"Directory exists. Contents of {skills_dir}:")
    try:
        # Check for root files
        root_files = list(skills_dir.glob("*.txt"))
        print(f"Found {len(root_files)} .txt files in root:")
        for f in root_files:
            print(f"  - {f.name}")
            
        # Check for categories
        categories = [d.name for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
        print(f"Found {len(categories)} categories:")
        for cat in categories:
            print(f"  - {cat}/")
            cat_files = list((skills_dir / cat).glob("*.txt"))
            print(f"    ({len(cat_files)} files)")
            
    except Exception as e:
        print(f"Exception during scan: {e}")

if __name__ == "__main__":
    test_discovery()