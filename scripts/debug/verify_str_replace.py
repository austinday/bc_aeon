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

def test_replacement(test_name, content, old_str, new_str):
    test_file = "test_target.txt"
    with open(test_file, "w") as f:
        f.write(content)
    
    worker = MockWorker()
    tool = StrReplaceTool(worker)
    
    result = tool.execute(file_path=test_file, old_str=old_str, new_str=new_str)
    
    with open(test_file, "r") as f:
        final_content = f.read()
    
    print(f"Test: {test_name}")
    print(f"Result: {result}")
    print(f"Final Content:\n{final_content}\n{'-'*20}")
    
    os.remove(test_file)
    return final_content

def main():
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    
    tests = [
        {
            "name": "Exact Match",
            "content": test_content,
            "old": "Line 2",
            "new": "REPLACED 2"
        },
        {
            "name": "Line-Number Stripped Match",
            "content": test_content,
            "old": "2: Line 2",
            "new": "REPLACED 2"
        },
        {
            "name": "L-Syntax Single Line",
            "content": test_content,
            "old": "L2",
            "new": "REPLACED 2"
        },
        {
            "name": "L-Syntax Range",
            "content": test_content,
            "old": "L2-L4",
            "new": "REPLACED 2-4"
        },
        {
            "name": "L-Syntax Full File",
            "content": test_content,
            "old": "L1-L5",
            "new": "ALL REPLACED"
        }
    ]
    
    all_passed = True
    for t in tests:
        res = test_replacement(t["name"], t["content"], t["old"], t["new"])
        # Simple check: did the content change?
        if res == t["content"]:
            print(f"FAILED: {t['name']} - No change detected")
            all_passed = False
        else:
            print(f"PASSED: {t['name']}")
            
    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")

if __name__ == "__main__":
    main()