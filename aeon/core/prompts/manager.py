import os
from pathlib import Path
from typing import List

# Define the base prompts directory
PROMPTS_DIR = Path(__file__).parent
TOOLS_PROMPTS_DIR = PROMPTS_DIR / "tools"
CATS_PROMPTS_DIR = PROMPTS_DIR / "categories"

def ensure_prompt_files(tool_names: List[str], category_paths: List[str]):
    """
    Ensures that every tool and category has a corresponding .txt file in the prompts directory.
    If a file does not exist, it creates an empty one.
    """
    TOOLS_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    CATS_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure tool prompt files exist
    for name in tool_names:
        file_path = TOOLS_PROMPTS_DIR / f"{name}.txt"
        if not file_path.exists():
            file_path.write_text("")
            
    # Ensure category prompt files exist
    for path in category_paths:
        # Replace slashes with underscores for flat file naming (e.g., 'image_tools/sub' -> 'image_tools_sub.txt')
        safe_name = path.replace('/', '_')
        file_path = CATS_PROMPTS_DIR / f"{safe_name}.txt"
        if not file_path.exists():
            file_path.write_text("")

def load_tool_prompt(name: str) -> List[str]:
    """
    Loads directives for a specific tool from its .txt file.
    Returns a list of strings (one per line).
    """
    file_path = TOOLS_PROMPTS_DIR / f"{name}.txt"
    if file_path.exists():
        content = file_path.read_text().strip()
        if not content:
            return []
        return content.splitlines()
    return []

def load_cat_prompt(path: str) -> List[str]:
    """
    Loads directives for a specific category from its .txt file.
    Returns a list of strings (one per line).
    """
    safe_name = path.replace('/', '_')
    file_path = CATS_PROMPTS_DIR / f"{safe_name}.txt"
    if file_path.exists():
        content = file_path.read_text().strip()
        if not content:
            return []
        return content.splitlines()
    return []