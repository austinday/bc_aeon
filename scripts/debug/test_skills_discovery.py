import os
from pathlib import Path
from aeon.core.skills.manager import SkillsManager

def test():
    print("--- Testing Skills Discovery ---")
    sm = SkillsManager()
    print(f"Skills base directory: {sm.base_dir}")
    print(f"Exists: {sm.base_dir.exists()}")
    
    try:
        categories = [d.name for d in sm.base_dir.iterdir() if d.is_dir()]
        print(f"Found categories: {categories}")
        
        for cat in categories:
            skills = sm.get_skills_in_category(cat)
            print(f"Category '{cat}': found {len(skills)} skills: {skills}")
            for skill in skills:
                content = sm.get_skill_content(cat, skill)
                print(f"  - {skill}: content length = {len(content) if content else 0}")
    except Exception as e:
        print(f"Error during discovery: {e}")

if __name__ == "__main__":
    test()