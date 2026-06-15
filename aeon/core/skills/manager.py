import os
import sys
from pathlib import Path
from typing import List, Optional
from aeon.core.paths import PROJECT_ROOT

class SkillsManager:
    """
    Manages the retrieval of skill protocols from the local filesystem.
    Prioritizes the local project root to avoid site-packages resolution issues.
    """
    def __init__(self):
        # Force resolution using aeon.core.paths to ensure we look at the source repo
        # and not a pip-installed site-packages directory which may lack txt files.
        self.base_dir = PROJECT_ROOT / "aeon" / "core" / "skills"
        
        # Fallback: If the above doesn't exist, try relative to cwd
        if not self.base_dir.exists():
            self.base_dir = Path(os.getcwd()) / "aeon" / "core" / "skills"

        # Final safety check: if we still can't find it, we are in a broken state
        if not self.base_dir.exists():
            # Log to stderr so it shows up in agent logs
            print(f"[CRITICAL] SkillsManager could not resolve skills directory. Tried: {self.base_dir}", file=sys.stderr)

    def get_skills_in_category(self, category_path: str) -> List[str]:
        """
        Returns a list of skill names (filenames without .txt) in the given category.
        """
        try:
            cat_dir = self.base_dir / category_path
            if not cat_dir.exists() or not cat_dir.is_dir():
                return []
            
            # Return all .txt files in the directory, removing the extension
            return [f.stem for f in cat_dir.glob("*.txt")]
        except Exception as e:
            print(f"[ERROR] SkillsManager.get_skills_in_category failed: {e}")
            return []

    def get_skill_content(self, category_path: str, skill_name: str) -> Optional[str]:
        """
        Reads the content of a specific skill protocol file.
        """
        try:
            skill_file = self.base_dir / category_path / f"{skill_name}.txt"
            if skill_file.exists():
                return skill_file.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"[ERROR] SkillsManager.get_skill_content failed: {e}")
        return None
