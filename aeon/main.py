import os, argparse, json, time, sys, subprocess, requests, fcntl, signal, atexit
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

LOCK_FILE_PATH = "/tmp/aeon_runtime.lock"
STARTUP_LOCK_PATH = "/tmp/aeon_brain_startup.lock"
MODEL_REGISTRY_PATH = "/tmp/aeon_model_registry.json"
MODEL_REGISTRY_LOCK_PATH = "/tmp/aeon_model_registry.lock"

# =============================================================================
# CLOUD MODEL DEFINITIONS
# =============================================================================
CLOUD_MODELS = [
    {
        'model': 'gemini-3.1-pro',
        'provider': 'vertex',
        'project_id': 'trout-cricket-9761108088181001',
        'context_limit': 2000000,
    },
    {
        'model': 'gemini-3.1-pro',
        'provider': 'vertex',
        'project_id': 'ai-ml-355015',
        'context_limit': 2000000,
    },
    {
        'model': 'grok-4-1-fast-reasoning',
        'provider': 'grok',
        'api_key_file': 'grok_api_key.txt',
        'base_url': 'https://api.x.ai/v1',
        'context_limit': 128000,
    },
    {
        'model': 'grok-4-1-fast-non-reasoning',
        'provider': 'grok',
        'api_key_file': 'grok_api_key.txt',
        'base_url': 'https://api.x.ai/v1',
        'context_limit': 128000,
    },
    {
        'model': 'gemini-3-pro-preview',
        'provider': 'gemini',
        'api_key_file': 'gemini_api_key.txt',
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
        'context_limit': 1000000,
    },
    {
        'model': 'gemini-flash-latest',
        'provider': 'gemini',
        'api_key_file': 'gemini_api_key.txt',
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
        'context_limit': 1000000,
    },
]

# =============================================================================
# LLAMA.CPP SERVED MODELS (GGUF, GPU+RAM hybrid, continuous batching)
# =============================================================================
LLAMACPP_MODELS = [
    {
        'model': 'Qwen3.5-397B-A17B-IQ4_XS',
        'label': 'Qwen3.5-397B MoE (IQ4_XS) MAX | Both GPUs full, ~9 t/s, 128k ctx | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8005/v1',
        'context_limit': 131072,
        'container_name': 'aeon_qwen397b',
        'start_script': 'start_qwen397b_max.sh',
        'health_port': 8005,
    },
    {
        'model': 'Qwen3.5-397B-A17B-IQ4_XS',
        'label': 'Qwen3.5-397B MoE (IQ4_XS) MEDIUM | GPU1 24GB free, ~6 t/s, 128k ctx | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8005/v1',
        'context_limit': 131072,
        'container_name': 'aeon_qwen397b',
        'start_script': 'start_qwen397b_medium.sh',
        'health_port': 8005,
    },
    {
        'model': 'Qwen3.5-397B-A17B-IQ4_XS',
        'label': 'Qwen3.5-397B MoE (IQ4_XS) LIGHT | GPU1 48GB free, ~3-5 t/s, 128k ctx | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8005/v1',
        'context_limit': 131072,
        'container_name': 'aeon_qwen397b',
        'start_script': 'start_qwen397b_light.sh',
        'health_port': 8005,
    },
    {
        'model': 'MiniMax-M2.5-Q5_K_S',
        'label': 'MiniMax-M2.5 MoE (Q5_K_S) | Req: GPU0 full + GPU1 partial, all on GPU, 128k ctx | Not Abliterated | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8013/v1',
        'context_limit': 131072,
        'container_name': 'aeon_minimax_m25',
        'start_script': 'start_minimax_m25_q8.sh',
        'health_port': 8013,
    },
    {
        'model': 'MiniMax-M2.5-Q5_K_S-Light',
        'label': 'MiniMax-M2.5 MoE (Q5_K_S) LIGHT | GPU1 ~48GB free, CPU Offload, 128k ctx | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8014/v1',
        'context_limit': 131072,
        'container_name': 'aeon_minimax_m25_q5_light',
        'start_script': 'start_minimax_m25_q5_light.sh',
        'health_port': 8014,
    },
    {
        'model': 'MiniMax-M2.5-Q6_K-Light',
        'label': 'MiniMax-M2.5 MoE (Q6_K) LIGHT | GPU1 ~48GB free, CPU Offload, 128k ctx | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8015/v1',
        'context_limit': 131072,
        'container_name': 'aeon_minimax_m25_q6_light',
        'start_script': 'start_minimax_m25_q6_light.sh',
        'health_port': 8015,
    },
    {
        'model': 'MiniMax-M2.5-Q8_0-Light',
        'label': 'MiniMax-M2.5 MoE (Q8_0) LIGHT | GPU1 ~48GB free, CPU Offload, 128k ctx | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8016/v1',
        'context_limit': 131072,
        'container_name': 'aeon_minimax_m25_q8_light',
        'start_script': 'start_minimax_m25_q8_light.sh',
        'health_port': 8016,
    },
    {
        'model': 'Qwen3-Coder-Next-Abliterated-Q8_0',
        'label': 'Qwen3-Coder-Next (Q8_0) | Req: One GPU0, no CPU offload, 256k ctx | Abliterated | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8007/v1',
        'context_limit': 262144,
        'container_name': 'aeon_qwen3_coder_q8',
        'start_script': 'start_qwen3_coder_q8.sh',
        'health_port': 8007,
    },
]

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

def warm_up_models(local_model_names):
    """Preload local models into VRAM by making initial requests."""
    if not local_model_names:
        return
    print("[SYSTEM] Warming up models (preloading to VRAM)...")
    models_to_warm = list(dict.fromkeys(local_model_names))
    
    for model in models_to_warm:
        try:
            print(f"[SYSTEM]   >> Loading {model}...", end='', flush=True)
            resp = requests.post(
                "http://localhost:8000/api/generate",
                json={"model": model, "prompt": "hello", "options": {"num_predict": 1}},
                timeout=300
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
        subprocess.run("docker ps -a -q --filter 'name=aeon_research' --filter 'name=aeon_vision' | xargs -r docker rm -f", 
                       shell=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception as e:
        print(f"[WARN] Cleanup timed out or failed: {e}")

# =============================================================================
# LLAMA.CPP SERVER LIFECYCLE
# =============================================================================

def is_llamacpp_model(config):
    """Check if a model config is a llama.cpp-served model."""
    return config and config.get('provider') == 'llamacpp'

def get_llamacpp_config(model_name):
    """Find llama.cpp model config by name."""
    for m in LLAMACPP_MODELS:
        if m['model'] == model_name:
            return m
    return None

def start_llamacpp_server(config):
    """Start the llama.cpp server container for a given model config."""
    container_name = config['container_name']
    port = config['health_port']
    
    # Check if already running and healthy
    if is_container_running(container_name):
        try:
            resp = requests.get(f'http://localhost:{port}/health', timeout=5)
            if resp.status_code == 200:
                print(f"[LLAMACPP] {container_name} already running and healthy.")
                return True
        except:
            pass
    
    script_name = config['start_script']
    script = Path(__file__).parent / 'scripts' / script_name
    if not script.exists():
        print(f"[LLAMACPP] ERROR: Start script not found: {script}")
        return False
    
    print(f"[LLAMACPP] Starting {config['model']} server (this may take several minutes for model loading)...")
    result = subprocess.run(['bash', str(script)], capture_output=False)
    if result.returncode != 0:
        print(f"[LLAMACPP] ERROR: Failed to start {container_name}")
        return False
    return True

def stop_llamacpp_server(config):
    """Stop the llama.cpp server container."""
    container_name = config['container_name']
    print(f"[LLAMACPP] Stopping {container_name}...")
    try:
        subprocess.run(['docker', 'stop', container_name], capture_output=True, timeout=30)
        subprocess.run(['docker', 'rm', container_name], capture_output=True, timeout=10)
        print(f"[LLAMACPP] {container_name} stopped and removed.")
    except Exception as e:
        print(f"[WARN] Failed to stop {container_name}: {e}")

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
        try:
            registry = json.load(open(MODEL_REGISTRY_PATH)) if os.path.exists(MODEL_REGISTRY_PATH) else {}
        except (json.JSONDecodeError, EOFError):
            print(f"[WARN] Registry corrupted, resetting: {MODEL_REGISTRY_PATH}")
            registry = {}
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
    """Unregister this agent's PID and unload Ollama models with no remaining users.
    Note: llama.cpp container lifecycle is handled separately in SessionManager.exit().
    """
    if not models:
        return
    pid = os.getpid()
    to_unload = []
    # Determine which models are llama.cpp (don't try Ollama unload for them)
    llamacpp_model_names = {m['model'] for m in LLAMACPP_MODELS}
    with open(MODEL_REGISTRY_LOCK_PATH, 'w') as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            registry = json.load(open(MODEL_REGISTRY_PATH)) if os.path.exists(MODEL_REGISTRY_PATH) else {}
        except (json.JSONDecodeError, EOFError):
            print(f"[WARN] Registry corrupted, resetting: {MODEL_REGISTRY_PATH}")
            registry = {}
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
        if model in llamacpp_model_names:
            print(f"[SYSTEM] Skipping Ollama unload for llama.cpp model '{model}' (container lifecycle handled separately).")
            continue
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

# =============================================================================
# UNIFIED MODEL MENU
# =============================================================================

def build_model_menu(local_models):
    """Build a unified menu of all available models (local + cloud + llamacpp)."""
    entries = []
    entries.append({'label': '--- Local Models ---', 'is_header': True})
    for m in local_models:
        entries.append({
            'model': m,
            'provider': 'local',
            'context_limit': 128000,
            'label': f'{m} | Req: One GPU0 | Unknown if Abliterated | Local/Ollama',
        })
    for lm in LLAMACPP_MODELS:
        entry = dict(lm)
        entries.append(entry)

    entries.append({'label': '--- API Models ---', 'is_header': True})
    vertex_models = []
    for cm in CLOUD_MODELS:
        entry = dict(cm)
        if cm.get('provider') == 'vertex':
            entry['label'] = f"Vertex AI - Gemini 3.1 Pro (Billing: {cm['project_id']})"
            vertex_models.append(entry)
        else:
            entry['label'] = f"{cm['model']} | Req: Internet | Unrestricted status depends on API | API/Cloud"
            entries.append(entry)
    
    if vertex_models:
        entries.append({'label': '--- Vertex AI Models ---', 'is_header': True})
        entries.extend(vertex_models)
        
    return entries

def select_model(menu_entries, label):
    """Display unified model menu and return selected model config."""
    print(f'\n[MENU] {label}')
    selectable = []
    for entry in menu_entries:
        if entry.get('is_header'):
            print(f" {entry['label']}")
        else:
            selectable.append(entry)
            print(f"  {len(selectable)}. {entry['label']}")
    while True:
        try:
            choice = input(f'Select Model (1-{len(selectable)}): ')
            if choice.isdigit() and 1 <= int(choice) <= len(selectable):
                return selectable[int(choice)-1]
        except (KeyboardInterrupt, EOFError): sys.exit(0)
        except: pass
        print('Invalid choice.')

def find_model_config(model_name, menu_entries):
    """Find a model config by name from the menu entries."""
    for entry in menu_entries:
        if entry['model'] == model_name:
            return entry
    return None

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
        self._llamacpp_configs = []  # llama.cpp model configs used by this agent

    def enter(self, strong_config=None, weak_config=None, skip_warmup=False):
        """Enter the session: coordinate startup, warm models, acquire locks.
        
        Only starts/warms the local brain if at least one selected model is local.
        Cloud-only configurations skip brain management entirely.
        llama.cpp models get their own container lifecycle.
        """
        # Determine which models are local Ollama (need brain + registry)
        local_models = []
        if strong_config and strong_config.get('provider') == 'local':
            local_models.append(strong_config['model'])
        if weak_config and weak_config.get('provider') == 'local':
            local_models.append(weak_config['model'])
        local_models = list(dict.fromkeys(local_models))  # deduplicate
        self._models_used = local_models

        # Determine which models are llama.cpp served
        llamacpp_configs = []
        for cfg in [strong_config, weak_config]:
            if cfg and is_llamacpp_model(cfg):
                if cfg not in llamacpp_configs:
                    llamacpp_configs.append(cfg)
        self._llamacpp_configs = llamacpp_configs

        needs_brain = len(local_models) > 0

        # --- PHASE 1: Startup Coordination (only if local Ollama models needed) ---
        if needs_brain:
            self.startup_lock = open(STARTUP_LOCK_PATH, 'w+')
            try:
                fcntl.flock(self.startup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                is_first_starter = True
                print("[SESSION] Acquired startup lock (first starter).")
            except BlockingIOError:
                print("[SESSION] Another agent is starting up, waiting...")
                fcntl.flock(self.startup_lock, fcntl.LOCK_SH)
                is_first_starter = False
                print("[SESSION] Startup complete, proceeding.")

            if is_first_starter:
                brain_started = start_local_brain_services()
                if brain_started and not skip_warmup:
                    warm_up_models(local_models)
                fcntl.flock(self.startup_lock, fcntl.LOCK_SH)
        else:
            print("[SESSION] No local Ollama models selected, skipping brain startup.")

        # --- PHASE 1b: Start llama.cpp servers (shared across agents via ref counting) ---
        for lcfg in llamacpp_configs:
            model_name = lcfg['model']
            register_models_for_agent([model_name])
            self._models_used.append(model_name)
            if not start_llamacpp_server(lcfg):
                print(f"[SESSION] WARNING: Failed to start llama.cpp server for {model_name}")

        # --- PHASE 2: Register local Ollama models for reference counting ---
        if local_models:
            register_models_for_agent(local_models)

        # --- PHASE 3: Runtime Lock ---
        self.runtime_lock = open(LOCK_FILE_PATH, 'w+')
        fcntl.flock(self.runtime_lock, fcntl.LOCK_SH)
        print("[SESSION] Acquired runtime lock (agent active).")

        # --- PHASE 4: Signal Handlers ---
        self._original_sigint = None
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._atexit_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully (SIGTERM only, not SIGINT)."""
        print(f"\n[SESSION] Received SIGTERM, cleaning up...")
        self.exit()
        sys.exit(0)

    def _atexit_handler(self):
        """Fallback cleanup on normal exit."""
        self.exit()

    def exit(self):
        """Exit the session: cleanup tools, release locks, maybe unload brain / stop containers."""
        if self._cleanup_done:
            return
        self._cleanup_done = True
        
        print("[SESSION] Exiting...")
        
        cleanup_transient_tools()
        
        if self._models_used:
            # Check which llamacpp models will have zero users after unregister
            llamacpp_to_stop = []
            for lcfg in self._llamacpp_configs:
                model_name = lcfg['model']
                # Peek at registry to see if we're the last user
                try:
                    with open(MODEL_REGISTRY_LOCK_PATH, 'w') as lock_fd:
                        fcntl.flock(lock_fd, fcntl.LOCK_SH)
                        registry = json.load(open(MODEL_REGISTRY_PATH)) if os.path.exists(MODEL_REGISTRY_PATH) else {}
                        pids = registry.get(model_name, [])
                        alive_pids = [p for p in pids if _pid_exists(p) and p != os.getpid()]
                        if not alive_pids:
                            llamacpp_to_stop.append(lcfg)
                except:
                    pass  # If we can't check, unregister will handle it

            unregister_models_for_agent(self._models_used)
            
            # Stop llama.cpp containers that have no remaining users
            for lcfg in llamacpp_to_stop:
                stop_llamacpp_server(lcfg)
        
        if self.runtime_lock:
            try:
                fcntl.flock(self.runtime_lock, fcntl.LOCK_UN)
                self.runtime_lock.close()
            except Exception as e:
                print(f"[WARN] Session cleanup error: {e}")
        
        if self.startup_lock:
            try:
                self.startup_lock.close()
            except: pass
        
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        
        print("[SESSION] Cleanup complete.")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable detailed LLM call logging to ~/')
    parser.add_argument('--strong', type=str, dest='model', help='Model name - skips menu')
    parser.add_argument('--model', type=str, help='Model name - skips menu (alias for --strong)')
    parser.add_argument('--weak', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--start', type=str, help='Initial objective to start immediately')
    parser.add_argument('--no-warmup', action='store_true', help='Skip model warmup (faster startup, slower first query)')
    args = parser.parse_args()

    # --- Enumerate local models (start brain if needed) ---
    local_models = []
    if is_container_running("aeon_brain_node"):
        local_models = get_ollama_models()
    else:
        print("[SYSTEM] Starting brain to enumerate local models...")
        start_local_brain_services()
        local_models = get_ollama_models()

    if not local_models:
        print("[WARN] No local models found via API.")
        local_models = []

    # --- Build unified model menu (local + cloud) ---
    menu = build_model_menu(local_models)

    # --- Select model (used for both planning and utility tasks) ---
    model_name = args.model or args.weak  # --weak kept for backward compat
    if model_name:
        strong_config = find_model_config(model_name, menu)
        if not strong_config:
            print(f"[ERROR] Model '{model_name}' not found.")
            print(f"  Available: {[e['model'] for e in menu if not e.get('is_header')]}")
            sys.exit(1)
    else:
        strong_config = select_model(menu, 'Select Model')

    weak_config = strong_config

    print(f"[CONFIG] Model: {strong_config['model']} ({strong_config['provider']})")

    session = SessionManager()
    session.enter(strong_config=strong_config, weak_config=weak_config, skip_warmup=args.no_warmup)

    try:
        llm_client = LLMClient(strong_config=strong_config, weak_config=weak_config)
        worker = Worker(llm_client=llm_client, debug_mode=args.debug)
        deps = {'llm_client': llm_client, 'worker': worker}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        worker.register_tools(tools)

        prov = strong_config['provider'].upper()
        print(f"\nAeon Ready (Model: {strong_config['model']} [{prov}], Debug: {args.debug})")
        
        if args.start:
            worker.run(args.start)
        
        while True:
            try:
                obj = input("> ")
                if obj.strip(): 
                    if obj.strip() in ['exit', 'quit']: break
                    worker.run(obj)
            except (KeyboardInterrupt, EOFError):
                print("\n")
                break
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        raise
    finally:
        session.exit()

if __name__ == "__main__": cli()
