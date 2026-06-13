from aeon.core.skills.manager import SkillsManager

def test_hardcoded_skill():
    print("[TEST] Initializing SkillsManager...")
    sm = SkillsManager()
    
    skill_path = "research/web_research"
    print(f"[TEST] Attempting to retrieve skill: {skill_path}")
    
    content = sm.get_skill_content(skill_path)
    
    if not content:
        print("[FAIL] No content returned for hard-coded skill.")
        return False
    
    print(f"[SUCCESS] Content retrieved. Length: {len(content)} chars")
    print("--- CONTENT START ---")
    print(content)
    print("--- CONTENT END ---")
    
    expected_snippet = "Comprehensive Web Research Protocol"
    if expected_snippet in content:
        print(f"[VERIFIED] Found expected snippet: '{expected_snippet}'")
        return True
    else:
        print(f"[FAIL] Expected snippet '{expected_snippet}' not found in content.")
        return False

if __name__ == "__main__":
    success = test_hardcoded_skill()
    if success:
        exit(0)
    else:
        exit(1)