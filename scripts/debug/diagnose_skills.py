import os
from pathlib import Path
from aeon.core.paths import PROJECT_ROOT
from aeon.core.skills.manager import SkillsManager

def test_discovery():
    print(f"Project Root: {PROJECT_ROOT}")
    sm = SkillsManager()
    print(f"Skills Manager base_dir: {sm.base_dir}")
    
    if not sm.base_dir.exists():
        print(f"CRITICAL: Skills directory does not exist at {sm.base_dir}")
        return

    print("\n--- Scanning for Root Skills ---")
    root_skills = [f.stem for f in sm.base_dir.glob("*.txt") if not f.name.startswith('__')]
    print(f"Found root skills: {root_skills}")

    print("\n--- Scanning for Categories ---")
    categories = [d.name for d in sm.base_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
    print(f"Found categories: {categories}")

    for cat in categories:
        skills = sm.get_skills_in_category(cat)
        print(f"Category '{cat}': {skills}")

if __name__ == "__main__":
    test_discovery()