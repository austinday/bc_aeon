import os
import argparse
import json
import time
import sys
import subprocess
import requests
import fcntl
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

# ANSI Colors for System Messages
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

def is_container_running(name):
    """Checks if a docker container exists and is running."""
    try:
        output = subprocess.check_output(
            ["docker", "ps", "-q", "-f", f"name={name}"], 
            stderr=subprocess.DEVNULL, 
            text=True
        ).strip()
        return bool(output)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def wait_for_service(name, url, container_name, timeout=600):
    """Polls HTTP endpoint. Aborts immediately if container dies."""
    print(f"Waiting for {name} to hydrate at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # 1. Check if container is still alive
        if not is_container_running(container_name):
            print(f"\n\n[FATAL] Container '{container_name}' died while loading!")
            print("--- CRASH LOGS ---")
            try:
                subprocess.run(["docker", "logs", "--tail", "20", container_name])
            except Exception:
                print("Could not fetch logs.")
            print("------------------")
            return False

        # 2. Check HTTP
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f" -> {name} is ONLINE.")
                return True
        except requests.exceptions.RequestException:
            pass
        
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(2)
    
    print(f"\nTimeout waiting for {name}.")
    return False

def hydrate_local_models():
    """Forces models into VRAM synchronously. Blocks until ready."""
    print(f"{C_YELLOW}[SYSTEM] Warming up models (Loading to VRAM)... Please wait.{C_RESET}")
    
    def send_warmup(url, model, name):
        print(f"  >> Loading {name} ({model})... ", end="", flush=True)
        start = time.time()
        try:
            # We use a long timeout (120s) because loading 40GB+ from disk takes time.
            # num_predict: 1 ensures it returns instantly once loaded.
            requests.post(url, 
                        json={
                            "model": model, 
                            "prompt": "warmup", 
                            "keep_alive": -1,
                            "options": {"num_predict": 1}
                        }, 
                        timeout=120)
            elapsed = time.time() - start
            print(f"{C_GREEN}Done ({elapsed:.1f}s){C_RESET}")
        except Exception as e:
            print(f"{C_YELLOW}Failed: {e}{C_RESET}")

    # Load sequentially or parallel. Sequential is safer for power spikes, 
    # but we can do parallel since we have dual cards. 
    # Let's do sequential to keep the log clean and deterministic.
    send_warmup("http://localhost:8000/api/generate", "deepseek-r1:70b", "Planner (GPU 0)")
    send_warmup("http://localhost:8001/api/generate", "qwen2.5:72b", "Executor (GPU 1)")
    
    print(f"{C_GREEN}[SYSTEM] Brain fully heated and ready.{C_RESET}")

def ensure_local_brain_running():
    """Ensures Local Brain is running, handling race conditions between multiple agents."""
    planner_url = "http://localhost:8000/v1/models"
    executor_url = "http://localhost:8001/v1/models"
    lock_file_path = "/tmp/aeon_brain_startup.lock"
    
    containers_already_up = False

    # 1. Fast Check & Container Verification
    if is_container_running("aeon_planner") and is_container_running("aeon_executor"):
        containers_already_up = True
        # Even if containers are up, we verify endpoints are responding
        try:
            r1 = requests.get(planner_url, timeout=0.5)
            r2 = requests.get(executor_url, timeout=0.5)
            if r1.status_code == 200 and r2.status_code == 200:
                print("[SYSTEM] Local Brain containers active and endpoints responsive.")
                hydrate_local_models() # Crucial: Always re-warm on start (Blocking)
                return
        except requests.exceptions.RequestException:
            print("[SYSTEM] Containers active but endpoints sleeping/starting...")

    # 2. Start Sequence (If not up)
    if not containers_already_up:
        print("\n[SYSTEM] Local Brain services not detected. Acquiring startup lock...")
        with open(lock_file_path, 'w') as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError:
                print("Another agent is currently starting the brain. Waiting for it to finish...")
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                print("Lock acquired. Checking status...")
                if is_container_running("aeon_planner"):
                    print("Services started by previous agent. Proceeding to wait loop.")
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    wait_for_service("Planner (DeepSeek)", planner_url, "aeon_planner")
                    wait_for_service("Executor (Qwen)", executor_url, "aeon_executor")
                    hydrate_local_models()
                    return

            print("Initiating auto-launch sequence...")
            base_path = Path(__file__).parent
            script_path = base_path / "scripts" / "start_brain.sh"
            
            if not script_path.exists():
                raise FileNotFoundError(f"Critical Error: Startup script not found at {script_path}")

            try:
                subprocess.run(["bash", str(script_path)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[FATAL] Failed to start infrastructure. Exit code: {e.returncode}")
                print("Check Docker logs manually: 'docker logs aeon_planner'")
                sys.exit(1)
            
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    # 3. Final Wait & Warmup
    print("[SYSTEM] Infrastructure initialized. Waiting for endpoints...")
    if not wait_for_service("Planner (DeepSeek)", planner_url, "aeon_planner"):
        sys.exit(1)
    if not wait_for_service("Executor (Qwen)", executor_url, "aeon_executor"):
        sys.exit(1)
        
    hydrate_local_models()
    print("\n[SYSTEM] Local Brain is fully operational.\n")

def unload_local_models():
    """Sends unload request to local Ollama instances to free VRAM."""
    print(f"\n{C_YELLOW}[SYSTEM] Shutting down. Unloading models from VRAM...{C_RESET}")
    
    # Planner (DeepSeek R1)
    try:
        requests.post("http://localhost:8000/api/generate", 
                      json={"model": "deepseek-r1:70b", "keep_alive": 0}, 
                      timeout=5)
    except Exception:
        pass

    # Executor (Qwen 2.5)
    try:
        requests.post("http://localhost:8001/api/generate", 
                      json={"model": "qwen2.5:72b", "keep_alive": 0}, 
                      timeout=5)
    except Exception:
        pass
    print(f"{C_GREEN}[SYSTEM] VRAM released. Goodbye.{C_RESET}")

def cli():
    """Command-line interface for the Aeon agent."""    
    try:
        parser = argparse.ArgumentParser(description='Aeon Agent CLI')
        parser.add_argument('--restore', type=str, help='Path to history file for restoration')
        parser.add_argument('--grok', action='store_true', help='Use xAI Grok models (Cloud)')
        parser.add_argument('--gemini', action='store_true', help='Use Google Gemini models')
        parser.add_argument('--gemini-flash', action='store_true', help='Use Google Gemini Flash')
        parser.add_argument('--local', action='store_true', help='Explicitly use Local Brain (Default)')
        args = parser.parse_args()

        # 1. Determine Provider
        if args.grok:
            provider = "grok"
        elif args.gemini_flash:
            provider = "gemini-flash"
        elif args.gemini:
            provider = "gemini"
        else:
            provider = "local"

        # 2. Strict Local Setup
        if provider == "local":
            ensure_local_brain_running()
            
        print(f"Initializing Aeon with provider: {provider}...")
        llm_client = LLMClient(provider=provider)
        
        # 3. Initialize Worker
        worker = Worker(llm_client=llm_client)
        worker.max_history_tokens = max(int(os.getenv("AEON_MAX_TOKENS", "30000")), 25000)

        # 4. Dependency Injection
        deps = {
            'llm_client': llm_client,
            'worker': worker,
            'max_tokens': worker.max_history_tokens 
        }

        # 5. Load Tools
        print("Loading tools...")
        tools = load_tools_from_directory(package_name="aeon.tools", dependencies=deps)
        worker.register_tools(tools)

        print(f"Aeon Agent is ready with {len(worker.tools)} tools loaded.")
        print("Type your objective. Type 'exit' or CTRL+d to quit. Use /clear to reset.")
        
        while True:
            try:
                objective = input("> ")
                if objective.strip() == '/clear':
                    print("Resetting agent...")
                    # Re-verify local brain on reset
                    if provider == "local": ensure_local_brain_running()
                    
                    llm_client = LLMClient(provider=provider)
                    worker = Worker(llm_client=llm_client)
                    deps['llm_client'] = llm_client
                    deps['worker'] = worker
                    tools = load_tools_from_directory("aeon.tools", deps)
                    worker.register_tools(tools)
                    print('Agent reset.')
                    continue
                    
                if objective.lower().strip() in ['exit', 'quit']:
                    break

                worker.run(objective=objective)
            except KeyboardInterrupt:
                print("Interrupted. Press CTRL+d to quit.")
                continue
            except EOFError:
                break
                
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    
    finally:
        # Cleanup on exit (normal, error, or CTRL+D)
        if 'provider' in locals() and provider == "local":
            unload_local_models()

if __name__ == "__main__":
    cli()
