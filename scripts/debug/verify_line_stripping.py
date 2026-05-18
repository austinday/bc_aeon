import os
from aeon.tools.file_io import StrReplaceTool

# Mock worker
class MockWorker:
    def __init__(self):
        self.open_files = {}
    def is_file_open(self, path): return path in self.open_files
    def update_open_file(self, path, content): self.open_files[path] = content
    def close_file(self, path): self.open_files.pop(path, None)

def test_line_stripping():
    worker = MockWorker()
    tool = StrReplaceTool(worker)
    
    test_file = "test_strip.txt"
    content = "line one\nline two\nline three"
    with open(test_file, "w") as f:
        f.write(content)
    
    # Simulate a SEARCH block copied from OpenFileTool output
    # "1: line one\n2: line two"
    search_str = "1: line one\n2: line two"
    replace_str = "new line one\nnew line two"
    
    # We use the internal _apply_single_replace for direct testing
    new_content, method, err = tool._apply_single_replace(
        test_file, test_file, content, search_str, replace_str
    )
    
    print(f"Result content:\n{new_content}")
    print(f"Method: {method}")
    
    expected = "new line one\nnew line two\nline three"
    assert new_content == expected, f"Expected {repr(expected)}, got {repr(new_content)}"
    print("SUCCESS: Line number stripping verified.")
    os.remove(test_file)

if __name__ == "__main__":
    try:
        test_line_stripping()
    except Exception as e:
        print(f"FAILURE: {e}")
        exit(1)