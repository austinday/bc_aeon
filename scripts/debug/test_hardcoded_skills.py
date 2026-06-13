import sys
import os

# Ensure we can import from the root
sys.path.append(os.getcwd())

try:
    from aeon.core.skills.manager import SkillsManager
    print("Successfully imported SkillsManager")
except Exception as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_skills():
    sm = SkillsManager()
    
    print("\n--- Internal State Inspection ---")
    # Check if HARDCODED_SKILLS exists on the class
    if hasattr(SkillsManager, 'HARDCODED_SKILLS'):
        print(f"Class HARDCODED_SKILLS keys: {list(SkillsManager.HARDCODED_SKILLS.keys())}")
    else:
        print("CRITICAL: SkillsManager class has NO HARDCODED_SKILLS attribute!")

    # Check if it exists on the instance
    if hasattr(sm, 'HARDCODED_SKILLS'):
        print(f"Instance HARDCODED_SKILLS keys: {list(sm.HARDCODED_SKILLS.keys())}")
    else:
        print("Instance has no HARDCODED_SKILLS attribute")

    test_keys = [
        "research/web_research",
        "research/web_research.txt",
        "web_research",
        "research/web_research/"
    ]

    print("\n--- Testing get_skill_content ---")
    for key in test_keys:
        content = sm.get_skill_content(key)
        print(f"Key: '{key}' -> Content Length: {len(content)}")
        if content:
            print(f"Content snippet: {content[:50]}...")
        else:
            print("Result: EMPTY")

if __name__ == "__main__":
    test_skills()