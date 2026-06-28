import importlib
import pkgutil
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any
from aeon.tools.base import BaseTool

# ANSI Colors for loud failures
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

def load_tools_from_directory(
    package_name: str = "aeon.tools",
    dependencies: Dict[str, Any] = None,
    verbose: bool = False,
    errors_out: List[str] = None,
) -> List[BaseTool]:
    """Dynamically loads tools, filtering out those marked as internal.

    Genuine breakage (a module that fails to import, or a tool whose deps are
    satisfied but whose __init__ raises) is reported to stderr and appended to
    ``errors_out`` if provided. Tools merely missing dependencies are skipped
    silently — that is the normal mechanism for optional/late-bound tools.
    """
    if dependencies is None: dependencies = {}
    found_tools = []
    errors: List[str] = errors_out if errors_out is not None else []

    def _report(msg: str):
        errors.append(msg)
        print(f"{C_YELLOW}[tool loader] {msg}{C_RESET}", file=sys.stderr)

    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        msg = f"CRITICAL: could not import tool package {package_name}: {e}"
        errors.append(msg)
        print(f"{C_RED}{msg}{C_RESET}", file=sys.stderr)
        return []

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
        except Exception as e:
            # A module that won't even import is a real bug, not an optional skip.
            _report(f"module '{module_name}' failed to import: {type(e).__name__}: {e}")
            if verbose:
                traceback.print_exc()
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != full_module_name:
                continue

            if issubclass(obj, BaseTool) and obj is not BaseTool:
                init_signature = inspect.signature(obj.__init__)
                init_params = {}
                missing_deps = False
                for param_name, param in init_signature.parameters.items():
                    if param_name == 'self': continue
                    if param_name in dependencies: init_params[param_name] = dependencies[param_name]
                    elif param.default == inspect.Parameter.empty:
                        missing_deps = True
                        break
                if missing_deps:
                    continue  # optional/late-bound tool — silent by design
                try:
                    tool_instance = obj(**init_params)
                    if not getattr(tool_instance, 'is_internal', False):
                        if any(t.name == tool_instance.name for t in found_tools):
                            continue
                        found_tools.append(tool_instance)
                except Exception as e:
                    # Deps were satisfied but construction crashed -> real bug.
                    _report(f"tool class '{name}' in '{module_name}' failed to instantiate: "
                            f"{type(e).__name__}: {e}")
                    if verbose:
                        traceback.print_exc()

    return found_tools