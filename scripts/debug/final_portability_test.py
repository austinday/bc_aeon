import os
import sys

# Ensure the current directory is in path if not installed globally, 
# though restart_aeon should have handled installation.
sys.path.append("/home/aday/bc_aeon")

try:
    from aeon.tools.browser import BrowserNavigateTool
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test():
    print("--- Portability Final Test ---")
    
    # 1. Change directory to /tmp to simulate running from anywhere
    print("Changing directory to /tmp...")
    try:
        os.chdir("/tmp")
        print(f"Current working directory: {os.getcwd()}")
    except Exception as e:
        print(f"Failed to change directory: {e}")
        sys.exit(1)
    
    # 2. Initialize the tool
    print("Initializing BrowserNavigateTool...")
    try:
        tool = BrowserNavigateTool()
    except Exception as e:
        print(f"Failed to initialize tool: {e}")
        sys.exit(1)
    
    # 3. Execute navigation
    print("Attempting to navigate to https://example.com...")
    try:
        # This will trigger ensure_browser_running() -> start_browser.sh
        result = tool.execute(url="https://example.com")
        print("\n--- Tool Result ---\n")
        print(result)
        print("\n------------------\n")
        
        if "BROWSER ACTION SUCCESS" in result:
            print("SUCCESS: Browser tool is fully portable and functional from /tmp!")
        else:
            print("FAILURE: Tool executed but did not return a success message.")
            sys.exit(1)
            
    except Exception as e:
        print(f"FAILURE: An exception occurred during tool execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test()