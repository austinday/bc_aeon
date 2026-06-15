import os
from pathlib import Path
from aeon.core.skills.manager import SkillsManager

def test_summary():
    print("--- Testing Skills Discovery Logic ---")
    try:
        sm = SkillsManager()
        skills_dir = Path(sm.base_dir).resolve()
        print(f"Scanning skills directory: {skills_dir}")
        
        if not skills_dir.exists():
            print(f"ERROR: Skills directory does not exist at {skills_dir}")
            return

        # Mimic main.py logic
        root_skills = [f.stem for f in skills_dir.glob("*.txt") if not f.name.startswith('__')]
        skill_categories = [d.name for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
        
        print(f"Root skills found: {root_skills}")
        print(f"Categories found: {skill_categories}")
        
        for cat in skill_categories:
            skills = sm.get_skills_in_category(cat)
            print(f"Category '{cat}' contains: {skills}")

    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_summary()