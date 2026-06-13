import os
import sys
from aeon.core.skills.manager import SkillsManager

def test_skills_manager():
    print("--- Testing SkillsManager ---")
    sm = SkillsManager()
    
    # Verify base_dir is set
    print(f"Base dir: {sm.base_dir}")
    if not sm.base_dir.exists():
        print("ERROR: base_dir does not exist!")
        return

    # Test 'research' category
    category = "research"
    print(f"Testing category: {category}")
    skills = sm.get_skills_in_category(category)
    print(f"Found skills: {skills}")
    
    if not skills:
        print(f"ERROR: No skills found in category '{category}'. Expected at least ['web_research']")
    else:
        for skill in skills:
            content = sm.get_skill_content(category, skill)
            print(f"Skill '{skill}' content length: {len(content)}")
            if len(content) == 0:
                print(f"ERROR: Skill '{skill}' is empty!")

if __name__ == "__main__":
    test_skills_manager()