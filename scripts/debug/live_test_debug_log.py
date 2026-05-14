import subprocess
import os
import time
import json
from pathlib import Path

def run_live_test():
    log_file = "live_test_reasoning.jsonl"
    if Path(log_file).exists():
        Path(log_file).unlink()

    # Use a cloud model to avoid local server startup issues and speed up the test
    # Based on main.py, 'gemini-flash-latest' is a fast, reliable option.
    model = "gemini-flash-latest"
    objective = "Say 'Log Test' and then complete the task."
    
    cmd = [
        "python3", "-m", "aeon.main",
        "--debug-log", log_file,
        "--model", model,
        "--start", objective
    ]

    print(f"Launching Aeon with debug log: {' '.join(cmd)}")
    
    try:
        # Use a generous timeout since LLM calls can take time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            env=os.environ.copy()
        )

        # Poll for the log file to be created and have content
        start_time = time.time()
        timeout = 120 
        while time.time() - start_time < timeout:
            if Path(log_file).exists() and Path(log_file).stat().st_size > 0:
                print(f"Success: Log file {log_file} created and contains data.")
                process.terminate()
                break
            
            # Print a bit of stdout to see progress
            # Note: This is tricky with Popen, so we just wait.
            time.sleep(2)
        else:
            print("Timeout reached: Log file not found or empty.")
            process.terminate()
            # Print output for debugging
            stdout, stderr = process.communicate()
            print("\n--- AEON STDOUT ---")
            print(stdout)
            print("\n--- AEON STDERR ---")
            print(stderr)
            exit(1)

        # Verify the content of the log
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                print("FAILED: Log file is empty.")
                exit(1)
            
            first_entry = json.loads(lines[0])
            required_fields = ["iteration", "thought", "intent", "updated_plan", "actions", "result"]
            missing = [f for f in required_fields if f not in first_entry]
            
            if missing:
                print(f"FAILED: Log entry missing fields: {missing}")
                print(f"Entry content: {first_entry}")
                exit(1)
            
            print("VERIFIED: Log entry contains all required reasoning trace fields.")
            print(f"First entry thought: {first_entry.get('thought')}")

    except Exception as e:
        print(f"An error occurred during the live test: {e}")
        exit(1)

    print("\nLIVE TEST SUCCESSFUL: The --debug-log feature is working as intended.")

if __name__ == "__main__":
    run_live_test()