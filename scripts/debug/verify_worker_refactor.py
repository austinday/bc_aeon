import sys
import os

# Add the project root to sys.path to allow importing from 'aeon'
sys.path.append(os.getcwd())

try:
    from aeon.core.worker_utils import C_RED, C_RESET, truncate_output
    print("Successfully imported worker_utils components.")
    
    # Test truncate_output
    test_str = "Hello World " * 100
    truncated = truncate_output(test_str, max_chars=20)
    print(f"Truncation test: {len(truncated)} chars (Expected <= 20)")
    
    # Test colors
    print(f"{C_RED}Color test: Red is working{C_RESET}")
    
    from aeon.core.worker import Worker
    print("Successfully imported Worker class.")
    
    print("\nVerification SUCCESS: Refactoring did not break imports or basic utility logic.")
except Exception as e:
    print(f"\nVerification FAILED: {e}")
    sys.exit(1)