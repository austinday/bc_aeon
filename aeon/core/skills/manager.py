import os
import sys
from pathlib import Path
from typing import List, Optional
from aeon.core.paths import PROJECT_ROOT


def _has_skill_content(directory: Path) -> bool:
    """Return True if `directory` exists and contains at least one .txt skill file."""
    try:
        if not directory.is_dir():
            return False
        return any(directory.rglob("*.txt"))
    except OSError:
        return False


class SkillsManager:
    """
    Manages retrieval of skill protocols from the filesystem.

    The skills directory is resolved relative to the INSTALLED package location
    (the directory this file lives in), so it works regardless of the current
    working directory and regardless of whether aeon is run from a source
    checkout or a pip install. Python resolves __file__ to wherever the package
    physically is, so this is the same mechanism the prompts loader already uses.

    Resolution order (first candidate that actually contains .txt files wins):
      1. AEON_SKILLS_DIR env override (explicit pointer; also useful for live
         editing of skills without reinstalling).
      2. Package-relative: the directory containing this file. Correct for both
         source checkouts and pip installs.
      3. cwd-relative: covers running from a source checkout while aeon was
         imported from a stale/incomplete site-packages copy.
      4. PROJECT_ROOT-relative: legacy last resort.
    """

    def __init__(self):
        package_skills = Path(__file__).resolve().parent

        candidates = []
        env_dir = os.environ.get("AEON_SKILLS_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.append(package_skills)
        candidates.append(Path.cwd() / "aeon" / "core" / "skills")
        candidates.append(PROJECT_ROOT / "aeon" / "core" / "skills")

        self.base_dir = None
        for candidate in candidates:
            if _has_skill_content(candidate):
                self.base_dir = candidate.resolve()
                break

        if self.base_dir is None:
            # No skill .txt files found anywhere. Still point at the package
            # directory (the correct location) so the agent keeps running; the
            # per-call methods below degrade gracefully to empty results. This
            # almost always means package data was not installed.
            self.base_dir = package_skills
            print(
                f"[WARNING] SkillsManager found no skill .txt files. Defaulting to "
                f"package directory: {self.base_dir}. If skills are missing, reinstall "
                f"aeon (pip install .) so packaged skills ship to site-packages, or set "
                f"the AEON_SKILLS_DIR environment variable to your skills directory.",
                file=sys.stderr,
            )

    def get_skills_in_category(self, category_path: str) -> List[str]:
        """
        Returns a list of skill names (filenames without .txt) in the given category.
        """
        try:
            cat_dir = self.base_dir / category_path
            if not cat_dir.exists() or not cat_dir.is_dir():
                return []
            return [f.stem for f in cat_dir.glob("*.txt")]
        except Exception as e:
            print(f"[ERROR] SkillsManager.get_skills_in_category failed: {e}", file=sys.stderr)
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
            print(f"[ERROR] SkillsManager.get_skill_content failed: {e}", file=sys.stderr)
        return None
