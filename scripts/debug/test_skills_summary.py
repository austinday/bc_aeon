import os
from pathlib import Path

def test_discovery():
    # Mimic the logic in main.py
    # We assume the current working directory is the project root
    skills_dir = Path("aeon/core/skills").resolve()
    print(f"Testing skills discovery at: {skills_dir}")
    
    if not skills_dir.exists():
        print(f"ERROR: Directory {skills_dir} does not exist!")
        return

    # Find all .txt files in the root of the skills directory
    root_skills = [f.stem for f in skills_dir.glob("*.txt") if not f.name.startswith('__')]
    print(f"Found root skills: {root_skills}")
    
    # Find all subdirectories (categories)
    skill_categories = [d.name for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
    print(f"Found categories: {skill_categories}")

    # Verify specific expected files
    expected = ["adding_skills", "adding_tools"]
    for exp in expected:
        if exp in root_skills:
            print(f"SUCCESS: Found expected skill {exp}")
        else:
            print(f"FAILURE: Missing expected skill {exp}")

if __name__ == "__main__":
    test_discovery()