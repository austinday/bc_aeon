from aeon.tools.categories import TOOL_CATEGORIES, get_all_category_paths, get_tools_in_category, TOP_LEVEL_TOOLS

def test():
    print("Testing Tool Categories...")
    print(f"TOP_LEVEL_TOOLS: {TOP_LEVEL_TOOLS}")
    
    paths = get_all_category_paths()
    print(f"Category Paths found: {paths}")
    
    for path in paths:
        tools = get_tools_in_category(path)
        print(f"Category {path} contains tools: {tools}")

if __name__ == "__main__":
    test()