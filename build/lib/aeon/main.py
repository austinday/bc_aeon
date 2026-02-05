import os, argparse, json, time, sys, subprocess, requests, fcntl, signal, atexit
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

LOCK_FILE_PATH = "/tmp/aeon_runtime.lock"
STARTUP_LOCK_PATH = "/tmp/aeon_brain_startup.lock"
MODEL_REGISTRY_PATH = "/tmp/aeon_model_registry.json"
MODEL_REGISTRY_LOCK_PATH = "/tmp/aeon_model_registry.lock"

def is_container_running(name):
    try: return bool(subprocess.check_output(["docker", "ps", "-q", "-f", f"name={name}"], stderr=subprocess.DEVNULL, text=True).strip())
    except: return False

def wait_for_service(name, port):
    print(f"Waiting for {name} (Port {port})...", end='', flush=True)
    start = time.time()
    while time.time() - start < 60:
        try:
            if requests.get(f"http://localhost:{port}/api/tags", timeout=1).status_code == 200: 
                print(" OK.")
                return True
        except: pass
        time.sleep(1)
        print(".", end='', flush=True)
    print(" Timeout!")
    return False

def start_local_brain_services():
    """Start the Ollama brain container if not already running."""
    if is_container_running("aeon_brain_node"):
        print("[SYSTEM] Brain node already running.")
        return True
    print("\n[SYSTEM] Booting Local Brain...")
    script = Path(__file__).parent / "scripts" / "start_brain.sh"
    subprocess.run(["bash", str(script)], check=True)
    return wait_for_service("Aeon Brain (Ollama)", 8000)

def warm_up_models(strong_model, weak_model):
    """Preload models into VRAM by making initial requests."""
    print("[SYSTEM] Warming up models (preloading to VRAM)...")
    models_to_warm = [m for m in [strong_model, weak_model] if m]
    # Deduplicate if same model used for both
    models_to_warm = list(dict.fromkeys(models_to_warm))
    
    for model in models_to_warm:
        try:
            print(f"[SYSTEM]  >> Loading {model}...", end='', flush=True)
            resp = requests.post(
                "http://localhost:8000/api/generate",
                json={"model": model, "prompt": "hello", "options": {"num_predict": 1}},
                timeout=300  # Models can take a while to load
            )
            if resp.status_code == 200:
                print(" OK.")
            else:
                print(f" Warning: Status {resp.status_code}")
        except requests.exceptions.Timeout:
            print(" Timeout (model may still be loading).")
        except Exception as e:
            print(f" Error: {e}")
    print("[SYSTEM] Model warmup complete.")

def cleanup_transient_tools():
    print("[SYSTEM] Cleaning up transient tool containers...")
    try:
        # Added timeout=5s to prevent hanging if Docker daemon is broken
        subprocess.run("docker ps -a -q --filter 'name=aeon_research' --filter 'name=aeon_vision' | xargs -r docker rm -f", 
                       shell=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception as e:
        print(f"[WARN] Cleanup timed out or failed: {e}")

def unload_local_brain():
    print("[SYSTEM] Last agent exiting. Releasing Brain VRAM...")
    try:
        resp = requests.get("http://localhost:8000/api/ps", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            if not models:
                print("[SYSTEM] No models loaded.")
                return
            for m in models:
                print(f"[SYSTEM] Unloading {m['name']}...")
                requests.post("http://localhost:8000/api/generate", json={"model": m['name'], "keep_alive": 0}, timeout=10)
            print("[SYSTEM] VRAM released.")
    except Exception as e:
        print(f"[WARN] Failed to release VRAM: {e}")

# =============================================================================
# MODEL REFERENCE COUNTING
# =============================================================================

def _cleanup_stale_pids(registry):
    """Remove PIDs that no longer exist. Returns (cleaned_registry, orphaned_models)."""
    cleaned = {}
    orphaned = []
    for model, pids in registry.items():
        alive = [p for p in pids if _pid_exists(p)]
        if alive:
            cleaned[model] = alive
        else:
            orphaned.append(model)
            print(f"[REGISTRY] Cleaning orphaned model '{model}' (dead PIDs: {pids})")
    return cleaned, orphaned

def _pid_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def register_models_for_agent(models):
    """Register this agent's PID for the given models."""
    if not models:
        return
    pid = os.getpid()
    with open(MODEL_REGISTRY_LOCK_PATH, 'w') as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        registry = json.load(open(MODEL_REGISTRY_PATH)) if os.path.exists(MODEL_REGISTRY_PATH) else {}
        registry, orphaned = _cleanup_stale_pids(registry)
        for model in models:
            if model not in registry:
                registry[model] = []
            if pid not in registry[model]:
                registry[model].append(pid)
                print(f"[REGISTRY] Registered PID {pid} for '{model}'")
        with open(MODEL_REGISTRY_PATH, 'w') as f:
            json.dump(registry, f, indent=2)
    for model in orphaned:
        print(f"[SYSTEM] Unloading orphaned model {model}...")
        requests.post("http://localhost:8000/api/generate", json={"model": model, "keep_alive": 0}, timeout=15)

def unregister_models_for_agent(models):
    """Unregister this agent's PID and unload models with no remaining users."""
    if not models:
        return
    pid = os.getpid()
    to_unload = []
    with open(MODEL_REGISTRY_LOCK_PATH, 'w') as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        registry = json.load(open(MODEL_REGISTRY_PATH)) if os.path.exists(MODEL_REGISTRY_PATH) else {}
        registry, orphaned = _cleanup_stale_pids(registry)
        to_unload.extend(orphaned)
        for model in models:
            if model in registry and pid in registry[model]:
                registry[model].remove(pid)
                print(f"[REGISTRY] Unregistered PID {pid} from '{model}'")
                if not registry[model]:
                    del registry[model]
                    if model not in to_unload:
                        to_unload.append(model)
                    print(f"[REGISTRY] Model '{model}' has no users, will unload")
                else:
                    print(f"[REGISTRY] Model '{model}' still has {len(registry[model])} user(s)")
        with open(MODEL_REGISTRY_PATH, 'w') as f:
            json.dump(registry, f, indent=2)
    for model in to_unload:
        print(f"[SYSTEM] Unloading {model}...")
        try:
            requests.post("http://localhost:8000/api/generate", json={"model": model, "keep_alive": 0}, timeout=15)
        except Exception as e:
            print(f"[WARN] Failed to unload {model}: {e}")

def get_ollama_models():
    try:
        resp = requests.get("http://localhost:8000/api/tags", timeout=1)
        if resp.status_code == 200:
            return sorted([m['name'] for m in resp.json().get('models', [])])
    except: pass
    return []

def select_model(models, label):
    print(f"\n[MENU] {label}")
    for i, m in enumerate(models):
        print(f" {i+1}. {m}")
    while True:
        try:
            choice = input(f"Select Model (1-{len(models)}): ")
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                return models[int(choice)-1]
        except (KeyboardInterrupt, EOFError): sys.exit(0)
        except: pass
        print("Invalid choice.")

class SessionManager:
    """Manages agent lifecycle with proper coordination for shared brain resources.
    
    Architecture:
    - Startup Lock (exclusive during startup, then shared): Ensures only one agent
      starts/warms the brain at a time. Others wait then proceed.
    - Runtime Lock (shared): All running agents hold this. Last one out gets exclusive
      and cleans up brain VRAM.
    """
    def __init__(self):
        self.runtime_lock = None
        self.startup_lock = None
        self._cleanup_done = False
        self._original_sigint = None
        self._original_sigterm = None
        self._models_used = []

    def enter(self, strong_model=None, weak_model=None, skip_warmup=False):
        """Enter the session: coordinate startup, warm models, acquire locks."""
        # Track models for reference counting (deduplicated)
        self._models_used = list(dict.fromkeys([m for m in [strong_model, weak_model] if m]))
        # --- PHASE 1: Startup Coordination ---
        # Use startup lock to ensure only one agent does startup/warmup
        self.startup_lock = open(STARTUP_LOCK_PATH, 'w+')
        
        try:
            # Try to get exclusive lock (non-blocking)
            fcntl.flock(self.startup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            is_first_starter = True
            print("[SESSION] Acquired startup lock (first starter).")
        except BlockingIOError:
            # Another agent is starting up - wait for them
            print("[SESSION] Another agent is starting up, waiting...")
            fcntl.flock(self.startup_lock, fcntl.LOCK_SH)  # Block until startup done
            is_first_starter = False
            print("[SESSION] Startup complete, proceeding.")
        
        if is_first_starter:
            # We're responsible for starting and warming the brain
            brain_started = start_local_brain_services()
            if brain_started and strong_model and not skip_warmup:
                warm_up_models(strong_model, weak_model)
            # Downgrade to shared lock - signals startup complete
            fcntl.flock(self.startup_lock, fcntl.LOCK_SH)
        
        # --- PHASE 2: Register models for reference counting ---
        register_models_for_agent(self._models_used)
        
        # --- PHASE 3: Runtime Lock ---
        # Acquire shared runtime lock (all active agents hold this)
        self.runtime_lock = open(LOCK_FILE_PATH, 'w+')
        fcntl.flock(self.runtime_lock, fcntl.LOCK_SH)
        print("[SESSION] Acquired runtime lock (agent active).")
        
        # --- PHASE 4: Signal Handlers ---
        # Install SIGTERM handler for external kill signals.
        # NOTE: We deliberately do NOT install SIGINT handler here.
        # Ctrl+C should propagate as KeyboardInterrupt to the worker loop,
        # which provides an interactive dialog for user to modify objectives.
        self._original_sigint = None  # Not intercepted - let KeyboardInterrupt propagate
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._atexit_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully (SIGTERM only, not SIGINT).
        
        SIGINT (Ctrl+C) is NOT handled here - it propagates as KeyboardInterrupt
        to the worker loop, which provides an interactive dialog for the user
        to modify objectives or provide guidance.
        """
        print(f"\n[SESSION] Received SIGTERM, cleaning up...")
        self.exit()
        sys.exit(0)

    def _atexit_handler(self):
        """Fallback cleanup on normal exit."""
        self.exit()

    def exit(self):
        """Exit the session: cleanup tools, release locks, maybe unload brain."""
        if self._cleanup_done:
            return
        self._cleanup_done = True
        
        print("[SESSION] Exiting...")
        
        # Always cleanup transient tools this agent might have spawned
        cleanup_transient_tools()
        
        # --- Unregister models (unloads if this was last user) ---
        if self._models_used:
            unregister_models_for_agent(self._models_used)
        
        # --- Runtime Lock Release ---
        if self.runtime_lock:
            try:
                fcntl.flock(self.runtime_lock, fcntl.LOCK_UN)
                self.runtime_lock.close()
            except Exception as e:
                print(f"[WARN] Session cleanup error: {e}")
        
        # --- Startup Lock Cleanup ---
        if self.startup_lock:
            try:
                self.startup_lock.close()
            except: pass
        
        # Restore original signal handlers
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        
        print("[SESSION] Cleanup complete.")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grok', action='store_true')
    parser.add_argument('--gemini', action='store_true')
    parser.add_argument('--gemini-flash', action='store_true')
    parser.add_argument('--local', action='store_true', help='Force local mode')
    parser.add_argument('--debug', action='store_true', help='Enable detailed LLM call logging to ~/')
    parser.add_argument('--strong', type=str, help='Model for Strong Node (Planner)')
    parser.add_argument('--weak', type=str, help='Model for Weak Node (Executor)')
    parser.add_argument('--start', type=str, help='Initial objective to start immediately')
    parser.add_argument('--no-warmup', action='store_true', help='Skip model warmup (faster startup, slower first query)')
    args = parser.parse_args()

    provider = "local"
    if args.grok: provider = "grok"
    elif args.gemini: provider = "gemini"
    elif args.gemini_flash: provider = "gemini-flash"

    local_strong = None
    local_weak = None

    session = SessionManager()
    
    if provider == "local":
        # For local mode, we need to select models BEFORE entering session
        # because session.enter() will warm them up
        
        # Check if brain is already running to get model list
        if is_container_running("aeon_brain_node"):
            models = get_ollama_models()
        else:
            # Temporarily start brain to get model list, then it will be
            # properly managed by session.enter()
            print("[SYSTEM] Starting brain to enumerate models...")
            start_local_brain_services()
            models = get_ollama_models()
        
        if not models:
            print("[WARN] No models found via API. Using defaults.")
            models = ["deepseek-r1:70b", "qwen2.5:72b"]
        
        if args.strong and args.weak:
            local_strong, local_weak = args.strong, args.weak
        else:
            local_strong = args.strong if args.strong else select_model(models, "Select Strong Model (Planner)")
            local_weak = args.weak if args.weak else select_model(models, "Select Weak Model (Executor)")
        
        print(f"[CONFIG] Strong: {local_strong} | Weak: {local_weak}")
        
        # Enter session - always pass models for registry tracking
        session.enter(strong_model=local_strong, weak_model=local_weak, skip_warmup=args.no_warmup)
    else:
        # Cloud providers don't need local brain management
        session.enter()

    try:
        llm_client = LLMClient(provider=provider, local_strong=local_strong, local_weak=local_weak)
        worker = Worker(llm_client=llm_client, debug_mode=args.debug)
        deps = {'llm_client': llm_client, 'worker': worker}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        worker.register_tools(tools)
        print(f"\nAeon Ready (Mode: {provider.upper()}, Debug: {args.debug})")
        
        if args.start:
            worker.run(args.start)
        
        while True:
            try:
                obj = input("> ")
                if obj.strip(): 
                    if obj.strip() in ['exit', 'quit']: break
                    worker.run(obj)
            except (KeyboardInterrupt, EOFError):
                print("\n")  # Clean line after ^C
                break
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        raise
    finally:
        session.exit()

if __name__ == "__main__": cli()
