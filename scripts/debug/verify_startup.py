import os
import sys
from pathlib import Path

# Mocking the environment to test the startup logic in main.py
def test_startup_summary():
    print("--- STARTUP SIMULATION ---")
    try:
        from aeon.core.skills.manager import SkillsManager
        sm = SkillsManager()
        skills_dir = Path(sm.base_dir).resolve()
        print(f"Checking skills directory: {skills_dir}")
        
        if skills_dir.exists():
            root_skills = [f.stem for f in skills_dir.glob("*.txt") if not f.name.startswith('__')]
            skill_categories = [d.name for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
            
            if root_skills or skill_categories:
                print("\n\033[92m[S-V-S-S-S] Loaded Skills:\033[0m", file=sys.stderr)
                if root_skills:
                    for skill in sorted(root_skills):
                        print(f"  - {skill}", file=sys.stderr)
                for cat in sorted(skill_categories):
                    skills = sm.get_skills_in_category(cat)
                    if skills:
                        print(f"  - {cat}/", file=sys.stderr)
                        for skill in sorted(skills):
                            print(f"    - {skill}", file=sys.stderr)
            else:
                print(f"\n[SYSTEM] No skill protocols found in: {skills_dir}", file=sys.stderr)
        else:
            print(f"\n[SYSTEM] Skills directory not found at: {skills_dir}", file=sys.stderr)
    except Exception as e:
        print(f"Error during simulation: {e}")

if __name__ == "__main__":
    test_startup_summary()