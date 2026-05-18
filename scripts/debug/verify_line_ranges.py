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

def test_range_replacement():
    worker = MockWorker()
    tool = StrReplaceTool(worker)
    
    test_file = "test_range_verify.txt"
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    with open(test_file, "w") as f:
        f.write(content)
    
    try:
        # Test 1: Single line replacement L2
        # We use the 'old_str' parameter directly for the range syntax
        res = tool.execute(test_file, old_str="L2", new_str="REPLACED 2\n")
        with open(test_file, "r") as f:
            curr = f.read()
        assert "Line 1\nREPLACED 2\nLine 3" in curr, f"Test 1 failed: {curr}"
        print("Test 1 (Single Line L2) passed.")

        # Reset file
        with open(test_file, "w") as f: f.write(content)

        # Test 2: Range replacement L2-L4
        res = tool.execute(test_file, old_str="L2-L4", new_str="REPLACED 2-4\n")
        with open(test_file, "r") as f:
            curr = f.read()
        assert "Line 1\nREPLACED 2-4\nLine 5" in curr, f"Test 2 failed: {curr}"
        print("Test 2 (Range L2-L4) passed.")

        # Reset file
        with open(test_file, "w") as f: f.write(content)

        # Test 3: Out of bounds
        res = tool.execute(test_file, old_str="L10-L12", new_str="X")
        assert "Error" in res and "out of bounds" in res, f"Test 3 failed: {res}"
        print("Test 3 (Out of bounds) passed.")

        # Test 4: Start > End
        res = tool.execute(test_file, old_str="L4-L2", new_str="X")
        assert "Error" in res and "greater than end line" in res, f"Test 4 failed: {res}"
        print("Test 4 (Start > End) passed.")

        # Test 5: Full file replacement L1-L5
        with open(test_file, "w") as f: f.write(content)
        res = tool.execute(test_file, old_str="L1-L5", new_str="ALL NEW")
        with open(test_file, "r") as f:
            curr = f.read()
        assert curr == "ALL NEW", f"Test 5 failed: {curr}"
        print("Test 5 (Full range) passed.")

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_range_replacement()
    print("All line-range tests passed successfully.")