#!/usr/bin/env python3
"""
Minimal smoke test for Aeon agent.
Verifies the codebase is importable and tools can load.
Exit code 0 = safe to restart. Non-zero = broken.
"""
import sys
import os
from pathlib import Path

# Ensure the project root is in sys.path so we test the local package, not a shadowed one
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

def main():
    errors = []

    # Test 1: Core imports
    try:
        from aeon.core.worker import Worker
        from aeon.core.llm import LLMClient
    except Exception as e:
        errors.append(f'Core import failed: {e}')
        print(f'[FAIL] Core imports: {e}', file=sys.stderr)

    # Test 2: Prompt loading
    try:
        from aeon.core import prompts
        assert prompts.CORE_DIRECTIVES, 'CORE_DIRECTIVES is empty'
        assert prompts.PRIMARY_AGENT_INSTRUCTIONS, 'PRIMARY_AGENT_INSTRUCTIONS is empty'
    except Exception as e:
        errors.append(f'Prompt loading failed: {e}')
        print(f'[FAIL] Prompts: {e}', file=sys.stderr)

    # Test 3: Tool discovery (no deps needed for basic import check)
    try:
        from aeon.tools.loader import load_tools_from_directory
        tools = load_tools_from_directory('aeon.tools', dependencies={}, verbose=False)
        tool_names = [t.name for t in tools]
        assert 'run_command' in tool_names, f'run_command missing from {tool_names}'
        assert 'task_complete' in tool_names, f'task_complete missing from {tool_names}'
    except Exception as e:
        errors.append(f'Tool loader failed: {e}')
        print(f'[FAIL] Tool loader: {e}', file=sys.stderr)

    # Test 4: Syntax check all .py files (skip junk directories)
    try:
        import py_compile
        from pathlib import Path
        aeon_root = Path(__file__).parent
        skip_dirs = {'.ipynb_checkpoints', '__pycache__', 'node_modules', '.git'}
        py_files = [
            pf for pf in aeon_root.rglob('*.py')
            if not any(part in skip_dirs for part in pf.parts)
        ]
        for pf in py_files:
            try:
                py_compile.compile(str(pf), doraise=True)
            except py_compile.PyCompileError as ce:
                errors.append(f'Syntax error: {ce}')
                print(f'[FAIL] Syntax: {ce}', file=sys.stderr)
    except Exception as e:
        errors.append(f'Syntax check failed: {e}')

    if errors:
        print(f'SMOKE TEST FAILED: {len(errors)} error(s)', file=sys.stderr)
        for err in errors:
            print(f'  - {err}', file=sys.stderr)
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
