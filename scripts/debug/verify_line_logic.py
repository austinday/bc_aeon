import os
from aeon.tools.file_io import StrReplaceTool

class MockWorker:
    def __init__(self):
        self.open_files = {}
    def is_file_open(self, path):
        return path in self.open_files
    def update_open_file(self, path, content):
        self.open_files[path] = content
    def close_file(self, path):
        self.open_files.pop(path, None)

def test_replacement(test_name, initial_content, old_str, new_str):
    print(f"Testing: {test_name}")
    test_file = "test_target.txt"
    with open(test_file, "w") as f:
        f.write(initial_content)
    
    worker = MockWorker()
    tool = StrReplaceTool(worker)
    
    # We use old_str and new_str directly (not patch) for simpler testing
    result = tool.execute(test_file, old_str=old_str, new_str=new_str)
    
    with open(test_file, "r") as f:
        final_content = f.read()
    
    os.remove(test_file)
    return result, final_content

def run_tests():
    test_cases = [
        {
            "name": "Exact Match",
            "initial": "line1\nline2\nline3",
            "old": "line2",
            "new": "REPLACED",
            "expected": "line1\nREPLACED\nline3"
        },
        {
            "name": "Line Number Stripping",
            "initial": "line1\nline2\nline3",
            "old": "2: line2",
            "new": "REPLACED",
            "expected": "line1\nREPLACED\nline3"
        },
        {
            "name": "Line Range Single",
            "initial": "line1\nline2\nline3",
            "old": "L2",
            "new": "REPLACED",
            "expected": "line1\nREPLACED\nline3"
        },
        {
            "name": "Line Range Multi",
            "initial": "line1\nline2\nline3\nline4",
            "old": "L2-L3",
            "new": "REPLACED",
            "expected": "line1\nREPLACED\nline4"
        },
        {
            "name": "Line Range Start",
            "initial": "line1\nline2\nline3",
            "old": "L1-L1",
            "new": "REPLACED",
            "expected": "REPLACED\nline2\nline3"
        },
        {
            "name": "Line Range End",
            "initial": "line1\nline2\nline3",
            "old": "L3-L3",
            "new": "REPLACED",
            "expected": "line1\nline2\nREPLACED"
        }
    ]

    passed = 0
    for tc in test_cases:
        res, actual = test_replacement(tc["name"], tc["initial"], tc["old"], tc["new"])
        if actual == tc["expected"]:
            print(f"  [PASS] {tc['name']}")
            passed += 1
        else:
            print(f"  [FAIL] {tc['name']}")
            print(f"    Expected: {repr(tc['expected'])}")
            print(f"    Actual:   {repr(actual)}")
            print(f"    Tool Result: {res}")
    
    print(f"\nPassed {passed}/{len(test_cases)} tests.")

if __name__ == "__main__":
    run_tests()