import os

def test_replace(tool_instance, file_path, old_str, new_str, expected_content, test_name):
    print(f"Testing {test_name}...", end=" ")
    # We simulate the tool's execute logic since we are testing the internal _apply_single_replace
    # But for a true test, we can just use the tool's public method if we mock the worker
    class MockWorker:
        def is_file_open(self, p): return False
        def update_open_file(self, p, c): pass
        def close_file(self, p): pass

    from aeon.tools.file_io import StrReplaceTool
    worker = MockWorker()
    tool = StrReplaceTool(worker)
    
    # Setup file
    with open(file_path, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
    
    result = tool.execute(file_path, old_str=old_str, new_str=new_str)
    
    with open(file_path, 'r') as f:
        actual_content = f.read()
    
    if actual_content == expected_content:
        print("PASSED")
        return True
    else:
        print(f"FAILED\nExpected:\n{expected_content}\nActual:\n{actual_content}\nResult: {result}\n")
        return False

def main():
    test_file = "test_replace_verify.txt"
    successes = 0
    total = 0

    tests = [
        {
            "name": "Exact Match",
            "old": "Line 2\nLine 3",
            "new": "Replacement 2-3",
            "expected": "Line 1\nReplacement 2-3\nLine 4\nLine 5"
        },
        {
            "name": "Line Number Stripping",
            "old": "2: Line 2\n3: Line 3",
            "new": "Replacement 2-3",
            "expected": "Line 1\nReplacement 2-3\nLine 4\nLine 5"
        },
        {
            "name": "L-Syntax Single Line",
            "old": "L2",
            "new": "Replacement 2",
            "expected": "Line 1\nReplacement 2\nLine 3\nLine 4\nLine 5"
        },
        {
            "name": "L-Syntax Range",
            "old": "L2-L3",
            "new": "Replacement 2-3",
            "expected": "Line 1\nReplacement 2-3\nLine 4\nLine 5"
        },
        {
            "name": "L-Syntax Full File",
            "old": "L1-L5",
            "new": "All Replaced",
            "expected": "All Replaced"
        }
    ]

    try:
        for t in tests:
            total += 1
            if test_replace(None, test_file, t["old"], t["new"], t["expected"], t["name"]):
                successes += 1
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

    print(f"\nTotal: {total}, Passed: {successes}, Failed: {total-successes}")
    if successes != total:
        exit(1)

if __name__ == "__main__":
    main()