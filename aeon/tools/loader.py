import importlib
import pkgutil
import inspect
import os
from types import ModuleType
from typing import List, Dict, Any

from aeon.tools.base import BaseTool

def load_tools_from_directory(
    package_name: str = "aeon.tools", 
    dependencies: Dict[str, Any] = None,
    verbose: bool = True
) -> List[BaseTool]:
    """
    Dynamically loads tools from the specified package directory.
    
    Args:
        package_name: The python package path to scan (e.g. 'aeon.tools')
        dependencies: A dictionary of objects available for injection 
                      (e.g. {'llm_client': llm, 'worker': worker})
        verbose: If True, prints loaded tools to stdout.
    """
    if dependencies is None:
        dependencies = {}
        
    found_tools = []
    
    # Import the package to get its file path
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        if verbose: print(f"Error importing package {package_name}: {e}")
        return []

    # Iterate over all modules in the package directory
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        
        try:
            module = importlib.import_module(full_module_name)
            
            # Scan module for classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it inherits from BaseTool and is not BaseTool itself
                if issubclass(obj, BaseTool) and obj is not BaseTool:
                    
                    # --- Dependency Injection Logic ---
                    init_signature = inspect.signature(obj.__init__)
                    init_params = {}
                    missing_deps = False

                    for param_name, param in init_signature.parameters.items():
                        if param_name == 'self':
                            continue
                        
                        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                            continue

                        if param_name in dependencies:
                            init_params[param_name] = dependencies[param_name]
                        elif param.default == inspect.Parameter.empty:
                            missing_deps = True
                            break
                    
                    if not missing_deps:
                        try:
                            tool_instance = obj(**init_params)
                            found_tools.append(tool_instance)
                            if verbose:
                                print(f"Loaded tool: {tool_instance.name}")
                        except Exception as e:
                            if verbose:
                                print(f"Failed to instantiate {name}: {e}")

        except ImportError as e:
            pass

    return found_tools