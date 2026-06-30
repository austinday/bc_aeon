import os
from pathlib import Path
from typing import Optional

def resolve_project_root() -> Path:
    """
    Resolves the install/source root of the aeon package itself.

    This is where aeon's CODE lives (prompts, scripts, self-improvement
    sources, logs). It is deliberately distinct from the *workspace* — the
    directory the user runs aeon from. The install root must NEVER be the
    workspace, and must never be hardcoded: it is derived from the location of
    the installed package.

    Logic:
    1. AEON_PROJECT_ROOT environment override (must contain setup.py).
    2. Walk up from this file looking for setup.py (editable / source checkout).
    3. Fall back to the package's install location, derived from this module's
       own path (parent of the `aeon` package dir). When pip-installed this is
       site-packages; in either case it is the real location of the code, not
       the user's current directory.
    """
    # 1. Environment Variable Override
    env_root = os.environ.get("AEON_PROJECT_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists() and (p / "setup.py").exists():
            return p

    # 2. Walk up from this file (aeon/core/paths.py) for a source checkout.
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "setup.py").exists():
            return parent

    # 3. Fall back to the installed package location, obtained from the package
    #    itself. paths.py is at <install_root>/aeon/core/paths.py, so the
    #    install root (the dir containing the `aeon` package) is two parents up
    #    from the `aeon` dir, i.e. parents[2] of this file. This keeps aeon
    #    portable: never the workspace, never a hardcoded path.
    return current_file.parents[2]

# Global constants for the session
PROJECT_ROOT = resolve_project_root()
# Runtime root is where the user is executing from, or a designated output dir
RUNTIME_ROOT = Path(os.getcwd()).resolve()

def get_project_root() -> Path:
    """Install/source root of the aeon package (where its code lives)."""
    return PROJECT_ROOT

def get_runtime_root() -> Path:
    return RUNTIME_ROOT

def get_workspace_root() -> Path:
    """The directory aeon was launched from — the user's current workspace.

    Resolved live from the process cwd so it tracks any chdir the harness
    performs. All user-facing assets and per-workspace bookkeeping live here,
    NOT in the aeon install dir.
    """
    return Path(os.getcwd()).resolve()

def get_output_dir() -> Path:
    """Directory for internal per-workspace agent bookkeeping (state, jobs,
    sub-agents). Distinct from user-facing assets, which go to the workspace
    root via resolve_output_path()."""
    return get_workspace_root() / "aeon_output"

def resolve_output_path(output_path: Optional[str], default_basename: str) -> Path:
    """Resolve where a generated asset (image/video/etc.) should be written.

    Rules:
      - Absolute output_path: honored exactly as given.
      - Relative output_path: resolved against the WORKSPACE root (the directory
        aeon was launched from), never the aeon install dir and never an
        implicit aeon_output/comfyui subdir.
      - Empty / None output_path: written as `default_basename` (used verbatim;
        the caller embeds any timestamp) at the workspace root.

    Returns an absolute Path. The caller is responsible for creating parent
    dirs. This is the single place that decides asset locations, so the
    behaviour is identical across every asset-producing tool.
    """
    workspace = get_workspace_root()
    if output_path and str(output_path).strip():
        p = Path(str(output_path).strip()).expanduser()
        if not p.is_absolute():
            p = workspace / p
        return p.resolve()
    # No path given: use the caller-provided default name at the workspace base.
    return (workspace / default_basename).resolve()
