from aeon.tools.loader import load_tools_from_directory

def main():
    print("Checking for generate_video tool...")
    tools = load_tools_from_directory()
    tool_names = [t.name for t in tools]
    
    if 'generate_video' in tool_names:
        print("SUCCESS: 'generate_video' tool was successfully loaded.")
    else:
        print(f"FAILURE: 'generate_video' not found in loaded tools. Found: {tool_names}")
        exit(1)

if __name__ == "__main__":
    main()