import sys
import os

# Ensure we can import from the project root
sys.path.insert(0, os.getcwd())

try:
    from aeon.core.skills.manager import SkillsManager
    print("Successfully imported SkillsManager")
    
    sm = SkillsManager()
    category = 'research'
    skills = sm.get_skills_in_category(category)
    
    print(f"Category: {category}")
    print(f"Skills found: {skills}")
    
    if skills and 'web_research' in skills:
        print("SUCCESS: 'web_research' found in 'research' category.")
    else:
        print("FAILURE: 'web_research' not found or category empty.")
        
    # Debug path info
    print(f"Skills Manager Base Dir: {getattr(sm, 'base_dir', 'N/A')}")
    
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()