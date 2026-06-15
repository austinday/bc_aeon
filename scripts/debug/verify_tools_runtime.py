import sys
import os

# Add current directory to path to import aeon
sys.path.insert(0, os.getcwd())

try:
    from aeon.tools.skills_manager_tool import ExpandSkillsCategory, CollapseSkillsCategory
    print("Successfully imported ExpandSkillsCategory and CollapseSkillsCategory")
    
    # Check if they have the execute method
    for cls in [ExpandSkillsCategory, CollapseSkillsCategory]:
        has_execute = hasattr(cls, 'execute')
        print(f"Class {cls.__name__} has execute method: {has_execute}")
        if has_execute:
            print(f"  - execute signature: {cls.execute}")

    # Simulate the manual registration in main.py
    # We don't need a full Worker/LLMClient, just check if we can create the objects
    # with dummy dependencies to see if they are still 'abstract'
    print("\nTesting instantiation with dummy deps...")
    try:
        t1 = ExpandSkillsCategory(worker=None, llm_client=None)
        print("ExpandSkillsCategory instantiated successfully")
    except Exception as e:
        print(f"ExpandSkillsCategory instantiation failed: {e}")

    try:
        t2 = CollapseSkillsCategory(worker=None, llm_client=None)
        print("CollapseSkillsCategory instantiated successfully")
    except Exception as e:
        print(f"CollapseSkillsCategory instantiation failed: {e}")

except Exception as e:
    print(f"Import error: {e}")