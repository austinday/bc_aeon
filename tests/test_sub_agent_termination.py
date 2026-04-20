import os
import signal
import subprocess
import time
from pathlib import Path
import sys

# Import the function to test
sys.path.append(os.getcwd())
from aeon.main import terminate_all_sub_agents

def test_termination():
    print("Starting sub-agent termination test...")
    
    # Setup: Create a dummy sub-agent directory and a dummy process
    test_dir = Path("aeon_output/sub_agents/test_agent_123")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Start a dummy process (sleep for 60s)
    proc = subprocess.Popen(["sleep", "60"])
    pid = proc.pid
    
    (test_dir / "pid.txt").write_text(str(pid))
    (test_dir / "status.txt").write_text("RUNNING")
    
    print(f"Spawned dummy process PID: {pid}")
    
    # Verify process is running
    try:
        os.kill(pid, 0)
        print("Verified: Process is running.")
    except OSError:
        print("Error: Process failed to start.")
        return False

    # Execute termination
    terminate_all_sub_agents()
    
    # Give it a moment to terminate
    time.sleep(1)
    
    # Verify process is dead
    # Use poll() instead of os.kill(pid, 0) because SIGKILL leaves a zombie 
    # until the parent polls it. os.kill(pid, 0) returns True for zombies.
    if proc.poll() is None:
        print(f"FAILED: Process {pid} is still alive.")
        proc.kill() # Cleanup
        return False
    else:
        print(f"SUCCESS: Process {pid} was terminated (Exit code: {proc.returncode}).")

    # Verify status file was updated
    status = (test_dir / "status.txt").read_text().strip()
    if status == "KILLED":
        print("SUCCESS: Status updated to KILLED.")
    else:
        print(f"FAILED: Status is {status}, expected KILLED.")
        return False

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    return True

if __name__ == "__main__":
    if test_termination():
        print("\nOverall Result: PASSED")
        sys.exit(0)
    else:
        print("\nOverall Result: FAILED")
        sys.exit(1)