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
                        except Exception as e:
                            if verbose: 
                                print(f"{C_RED}FAILED to initialize tool {name}: {e}{C_RESET}")
        except Exception as e:
            if verbose: 
                print(f"{C_RED}ERROR loading module {module_name}: {e}{C_RESET}")

    if verbose and found_tools:
        print("\nLoaded Tools:")
        try:
            from aeon.tools.categories import get_all_categorized_tools, TOOL_CATEGORIES, TOP_LEVEL_TOOLS
            categorized = get_all_categorized_tools()
            loaded_names = {t.name for t in found_tools}
            
            def _get_model_str(tool_name):
                t = next((x for x in found_tools if x.name == tool_name), None)
                if t and getattr(t, 'underlying_model', None):
                    return f" \033[90m[{t.underlying_model}]\033[0m"
                return ""

            for t in sorted(found_tools, key=lambda x: x.name):
                if t.name in TOP_LEVEL_TOOLS or t.name not in categorized:
                    print(f"  - {t.name}{_get_model_str(t.name)}")
            
            def _print_category(categories, depth=1):
                indent = "  " * depth
                for name, cat in categories.items():
                    cat_tools = [t for t in cat.get('tools', []) if t in loaded_names]
                    has_subcats = bool(cat.get('subcategories'))
                    if cat_tools or has_subcats:
                        print(f"{indent}- {name}/")
                        for tool_name in cat_tools:
                            print(f"{indent}  - {tool_name}{_get_model_str(tool_name)}")
                        if has_subcats:
                            _print_category(cat['subcategories'], depth + 1)
            
            if TOOL_CATEGORIES:
                _print_category(TOOL_CATEGORIES)
        except Exception:
            for t in found_tools:
                model_str = f" \033[90m[{t.underlying_model}]\033[0m" if getattr(t, 'underlying_model', None) else ""
                print(f"  - {t.name}{model_str}")

    return found_tools
