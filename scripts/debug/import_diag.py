import sys
import os
import traceback

print(f"Python version: {sys.version}")
print(f"Current Working Directory: {os.getcwd()}")
print("\n--- sys.path ---")
for p in sys.path:
    print(p)

def try_import(module_name):
    print(f"\nAttempting to import: {module_name}")
    try:
        mod = __import__(module_name, fromlist=[''])
        print(f"SUCCESS: {module_name} imported from {getattr(mod, '__file__', 'Unknown')}")
        return mod
    except Exception as e:
        print(f"FAILED: {module_name} - {e}")
        traceback.print_exc()
        return None

# Test 1: Base package
aeon = try_import('aeon')

# Test 2: Core worker
worker = try_import('aeon.core.worker')

# Test 3: Prompts
prompts = try_import('aeon.core.prompts')
if prompts:
    print(f"Checking for CORE_DIRECTIVES in aeon.core.prompts: {'CORE_DIRECTIVES' in dir(prompts)}")
    if not hasattr(prompts, 'CORE_DIRECTIVES'):
        print(f"Attributes of aeon.core.prompts: {dir(prompts)}")

# Test 4: Check for multiple aeon installations
import site
print("\n--- Site Packages ---")
print(site.getsitepackages())