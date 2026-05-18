import sys
import os

try:
    from aeon.core.session import SessionManager, CLOUD_MODELS, LLAMACPP_MODELS
    from aeon.core.worker import Worker
    from aeon.core.worker_utils import truncate_output
    print("Successfully imported all refactored modules.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_truncate():
    test_str = "Hello World" * 100
    truncated = truncate_output(test_str, 20)
    print(f"Truncate test: {'Success' if len(truncated) <= 20 else 'Failed'}")

if __name__ == "__main__":
    test_truncate()
    print("Verification script completed successfully.")