import importlib
import pkgutil
import inspect
import os
from typing import List, Dict, Any
from aeon.tools.base import BaseTool

# ANSI Colors for loud failures
C_RED = '\033[91m'
C_RESET = '\033[0m'

def load_tools_from_directory(
    package_name: str = "aeon.tools", 
    dependencies: Dict[str, Any] = None,
    verbose: bool = True
) -> List[BaseTool]:
    """Dynamically loads tools, filtering out those marked as internal."""
    if dependencies is None: dependencies = {}
    found_tools = []
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        if verbose: 
            print(f"{C_RED}CRITICAL ERROR: Could not import tool package {package_name}: {e}{C_RESET}")
        return []

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Ensure we only load tools defined in this module (prevents duplicates from imports)
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
                    if not missing_deps:
                        try:
                            tool_instance = obj(**init_params)
                            # Only add to main toolbox if not marked internal
                            if not getattr(tool_instance, 'is_internal', False):
                                # Dedup check by name
                                if any(t.name == tool_instance.name for t in found_tools):
                                    if verbose: print(f"Skipping duplicate tool '{tool_instance.name}' found in {module_name}")
                                    continue
                                
                                found_tools.append(tool_instance)
                                if verbose: print(f"Loaded tool: {tool_instance.name}")
                        except Exception as e:
                            if verbose: 
                                print(f"{C_RED}FAILED to initialize tool {name}: {e}{C_RESET}")
        except Exception as e:
            if verbose: 
                print(f"{C_RED}ERROR loading module {module_name}: {e}{C_RESET}")
    return found_tools