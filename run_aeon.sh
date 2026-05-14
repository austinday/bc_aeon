#!/bin/bash
# run_aeon.sh - Launcher for Aeon with crash reporting

# Ensure we are in the project root
PROJECT_ROOT="$(pwd)"
cd "$PROJECT_ROOT"

# Run the agent using the module path to ensure correct imports
# Pass all arguments through to the agent
python3 -m aeon.main "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "\n\033[91m[CRASH DETECTED] Aeon exited with code $EXIT_CODE\033[0m"
    echo -e "\033[93m--- LAST 100 LINES OF aeon.log ---\033[0m"
    if [ -f "aeon.log" ]; then
        tail -n 100 aeon.log
    else
        echo "Log file aeon.log not found."
    fi
    echo -e "\033[93m----------------------------------\033[0m"
fi

exit $EXIT_CODE