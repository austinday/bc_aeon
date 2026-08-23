import logging
import os
import sys
from pathlib import Path
from aeon.core.paths import get_workspace_root

def get_logger():
    logger = logging.getLogger('aeon')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Under the test suite, don't write to the real aeon.log: the tests
        # exercise the JSON-repair/fallback paths with synthetic garbage, and
        # their warnings are indistinguishable from real production failures in
        # the log. run_tests.sh sets AEON_NO_FILE_LOG=1; pytest is auto-detected.
        if (os.environ.get("AEON_NO_FILE_LOG") == "1"
                or "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules):
            file_handler = logging.NullHandler()
        else:
            # Log into the current workspace (where aeon was launched), not the
            # install dir — site-packages may be read-only, and run_aeon.sh tails
            # aeon.log from the workspace CWD.
            log_file = get_workspace_root() / 'aeon.log'
            file_handler = logging.FileHandler(log_file)
        # Console handler
        console_handler = logging.StreamHandler()
        # Only show ERROR or CRITICAL messages on the console for a cleaner user experience.
        console_handler.setLevel(logging.ERROR)
        # Formatter with timestamp, name, level, message for the file
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        # Simpler formatter for the console
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger