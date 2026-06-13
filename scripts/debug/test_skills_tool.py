import sys
import os

# Add the project root to sys.path to allow importing from aeon
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from aeon.tools.skills_manager_tool import ExpandSkillsCategory
    print("Successfully imported ExpandSkillsCategory")
    
    # Initialize the tool
    tool = ExpandSkillsCategory()
    
    # Test expanding the 'research' category
    print("Testing expand_skills_category('research')...")
    result = tool.execute({"category_path": "research"})
    print(f"Result: {result}")
    
    if "web_research" in result:
        print("SUCCESS: Found 'web_research' in the research category.")
    else:
        print("FAILURE: 'web_research' not found in result.")
        
except Exception as e:
    print(f"Import or Execution failed: {e}")
    import traceback
    traceback.print_exc()