import importlib
import pkgutil
import inspect
import os
import sys
from pathlib import Path

# Add current directory to path to ensure we load the local version of aeon
sys.path.insert(0, os.getcwd())

try:
    from aeon.tools.base import BaseTool
    from aeon.tools.loader import load_tools_from_directory
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def trace_loading():
    print("=== STARTING TOOL LOADING TRACE ===")
    package_name = "aeon.tools"
    dependencies = {'llm_client': None, 'worker': None} # Mock deps
    
    try:
        package = importlib.import_module(package_name)
        print(f"Successfully imported package: {package_name}")
    except Exception as e:
        print(f"Failed to import package {package_name}: {e}")
        return

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        print(f"\nChecking module: {full_module_name}")
        try:
            module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != full_module_name:
                    continue
                
                if issubclass(obj, BaseTool) and obj is not BaseTool:
                    print(f"  [FOUND] BaseTool class: {name}")
                    
                    # Trace the signature check
                    init_signature = inspect.signature(obj.__init__)
                    print(f"    Signature: {init_signature}")
                    
                    init_params = {}
                    missing_deps = False
                    for param_name, param in init_signature.parameters.items():
                        if param_name == 'self': continue
                        if param_name in dependencies: 
                            init_params[param_name] = dependencies[param_name]
                        elif param.default == inspect.Parameter.empty:
                            print(f"    [REJECTED] Missing dependency: {param_name}")
                            missing_deps = True
                            break
                    
                    if not missing_deps:
                        try:
                            tool_instance = obj(**init_params)
                            print(f"    [SUCCESS] Instance created: {tool_instance.name}")
                            if getattr(tool_instance, 'is_internal', False):
                                print(f"    [INFO] Tool marked as internal")
                        except Exception as e:
                            print(f"    [ERROR] Initialization failed: {e}")
                else:
                    pass
        except Exception as e:
            print(f"  [ERROR] Failed to process module {module_name}: {e}")

    print("\n=== TRACE COMPLETE ===")

if __name__ == "__main__":
    trace_loading()