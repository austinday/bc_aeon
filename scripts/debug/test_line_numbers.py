import os
import sys
from aeon.tools.file_io import StrReplaceTool
from aeon.tools.base import BaseTool

# Mock Worker for the tool
class MockWorker:
    def __init__(self):
        self.open_files = {}
    def is_file_open(self, path): return path in self.open_files
    def update_open_file(self, path, content): self.open_files[path] = content
    def close_file(self, path): 
        if path in self.open_files: del self.open_files[path]
        return True

def test_replacement():
    test_file = "test_target.txt"
    content = "Hello World\nThis is a test\nLine three\nLine four"
    with open(test_file, "w") as f:
        f.write(content)

    worker = MockWorker()
    tool = StrReplaceTool(worker)

    # Scenario 1: Exact match without line numbers (should still work)
    print("Testing Scenario 1: Exact match...")
    res1 = tool.execute(file_path=test_file, old_str="This is a test", new_str="This is a verified test")
    print(f"Result 1: {res1}")
    
    with open(test_file, "r") as f:
        curr = f.read()
    assert "This is a verified test" in curr
    print("Scenario 1 Passed.")

    # Reset file
    with open(test_file, "w") as f:
        f.write(content)

    # Scenario 2: Match with line numbers (simulating copy-paste from open_file)
    # The user sees:
    # 1: Hello World
    # 2: This is a test
    # 3: Line three
    # 4: Line four
    search_str = "2: This is a test"
    print(f"\nTesting Scenario 2: Line-numbered match ('{search_str}')...")
    res2 = tool.execute(file_path=test_file, old_str=search_str, new_str="Line 2 replaced")
    print(f"Result 2: {res2}")

    with open(test_file, "r") as f:
        curr = f.read()
    assert "Line 2 replaced" in curr
    print("Scenario 2 Passed.")

    # Scenario 3: Multi-line match with line numbers
    with open(test_file, "w") as f:
        f.write(content)
    
    search_multi = "2: This is a test\n3: Line three"
    print(f"\nTesting Scenario 3: Multi-line line-numbered match...")
    res3 = tool.execute(file_path=test_file, old_str=search_multi, new_str="Multi-line replaced")
    print(f"Result 3: {res3}")

    with open(test_file, "r") as f:
        curr = f.read()
    assert "Multi-line replaced" in curr
    print("Scenario 3 Passed.")

    os.remove(test_file)

if __name__ == "__main__":
    try:
        test_replacement()
        print("\nALL TESTS PASSED SUCCESSFULLY")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)