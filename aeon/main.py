import os, argparse, json, time, sys, subprocess, requests, fcntl, signal, atexit
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

LOCK_FILE_PATH = "/tmp/aeon_runtime.lock"
RESTART_STATE_PATH = f"/tmp/aeon_restart_state_{os.getpid()}.json"
RESTART_BACKUP_PATH = f"/tmp/aeon_restart_backup_{os.getpid()}.tar.gz"
STARTUP_LOCK_PATH = "/tmp/aeon_brain_startup.lock"
MODEL_REGISTRY_PATH = "/tmp/aeon_model_registry.json"
MODEL_REGISTRY_LOCK_PATH = "/tmp/aeon_model_registry.lock"

# =============================================================================
# CLOUD MODEL DEFINITIONS
# =============================================================================
CLOUD_MODELS = [
    {
        'model': 'gemini-3.1-pro-preview',
        'provider': 'vertex',
        'project_id': 'trout-cricket-9761108088181001',
        'context_limit': 2000000,
    },
    {
        'model': 'gemini-3.1-pro-preview',
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
        'model': 'Qwen3.6-35B-A3B-Uncensored',
        'family': 'Qwen3.6',
        'label': 'Qwen3.6-35B-A3B-Uncensored      | GPU0: 100%, GPU1: 0%     | ~?? t/s | 256k ctx | Abliterated: Yes | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8009/v1',
        'context_limit': 262144,
        'container_name': 'aeon_qwen36_35b',
        'start_script': 'start_qwen36_35b.sh',
        'health_port': 8009,
    },
    {
        'model': 'Gemma-4-31B-MTP-Q8_0',
        'family': 'Gemma-4',
        'label': 'Gemma-4-31B Native MTP (Atomic) | GPU0: 96GB, GPU1: 0%     | ~100+ t/s | 16k ctx  | Abliterated: Yes | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8013/v1',
        'context_limit': 16384,
        'container_name': 'aeon_gemma4_mtp',
        'start_script': 'start_gemma4_mtp.sh',
        'health_port': 8013,
    },
    {
        'model': 'Gemma-4-31B-Speculative-Q8_0',        'family': 'Gemma-4',
        'label': 'Gemma-4-31B Native MTP Cluster  | GPU0: 96GB, GPU1: 48GB   | ~80 t/s | 256k ctx | Abliterated: Yes | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8008/v1',
        'context_limit': 262144,
        'container_name': 'aeon_gemma_lb',
        'additional_containers': ['aeon_gemma4_node0', 'aeon_gemma4_node1'],
        'start_script': 'start_gemma4_speculative.sh',
        'health_port': 8008,
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
    env = os.environ.copy()
    env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    subprocess.run(["bash", str(script)], check=True, env=env)
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
        # Clean standard transient tools
        subprocess.run("docker ps -a -q --filter 'name=aeon_research' --filter 'name=aeon_vision' | xargs -r docker rm -f", 
                       shell=True, stderr=subprocess.DEVNULL, timeout=5)
        
        # Safely evaluate container cleanup using registry
        import fcntl
        my_pid = os.getpid()
        
        def _safe_cleanup(registry_path, lock_path, container_name, cleanup_callback=None):
            try:
                with open(lock_path, 'w') as lock_fd:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    if os.path.exists(registry_path):
                        with open(registry_path, 'r') as f:
                            active_pids = json.load(f)
                        
                        other_alive_pids = []
                        for p in active_pids:
                            if p == my_pid: continue
                            try:
                                os.kill(p, 0)
                                other_alive_pids.append(p)
                            except OSError:
                                pass
                                
                        if cleanup_callback:
                            cleanup_callback()
                                
                        if not other_alive_pids:
                            subprocess.run(["docker", "rm", "-f", container_name], stderr=subprocess.DEVNULL)
            except:
                pass
        
        _safe_cleanup("/tmp/aeon_comfyui_registry.json", "/tmp/aeon_comfyui_registry.lock", "aeon_comfyui")
        
        def _close_browser_session():
            try:
                requests.post("http://localhost:8030/close_session", json={"session_id": str(my_pid)}, timeout=2)
            except:
                pass
                
        _safe_cleanup("/tmp/aeon_browser_registry.json", "/tmp/aeon_browser_registry.lock", "aeon_browser", _close_browser_session)
        
    except Exception as e:
        print(f"[WARN] Cleanup timed out or failed: {e}")

# =============================================================================
# LLAMA.CPP SERVER LIFECYCLE
# =============================================================================

def is_llamacpp_model(config):
    """Check if a model config is a container-served model (llama.cpp or vLLM)."""
    return config and config.get('provider') in ['llamacpp', 'vllm']

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
    env = os.environ.copy()
    env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    result = subprocess.run(['bash', str(script)], capture_output=False, env=env)
    if result.returncode != 0:
        print(f"[LLAMACPP] ERROR: Failed to start {container_name}")
        return False
    return True

def stop_llamacpp_server(config):
    """Stop the llama.cpp server container(s)."""
    containers = [config['container_name']] + config.get('additional_containers', [])
    for container_name in containers:
        print(f"[LLAMACPP] Stopping {container_name}...")
        try:
            subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=30)
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
        # Explicitly check for zombies (Z state) in Linux
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat_content = f.read().split()
                if len(stat_content) > 2 and stat_content[2] == 'Z':
                    return False
        except FileNotFoundError:
            return False
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
        lcfg = get_llamacpp_config(model)
        if lcfg:
            print(f"[SYSTEM] Stopping orphaned llama.cpp cluster for {model}...")
            stop_llamacpp_server(lcfg)
        else:
            print(f"[SYSTEM] Unloading orphaned Ollama model {model}...")
            try:
                requests.post("http://localhost:8000/api/generate", json={"model": model, "keep_alive": 0}, timeout=15)
            except: pass

def unregister_models_for_agent(models):
    """Unregister this agent's PID and unload models with no remaining users."""
    if not models:
        return
    pid = os.getpid()
    to_unload = []
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
    for model in set(to_unload):
        lcfg = get_llamacpp_config(model)
        if lcfg:
            print(f"[SYSTEM] Stopping llama.cpp cluster for {model}...")
            stop_llamacpp_server(lcfg)
        else:
            print(f"[SYSTEM] Unloading Ollama model {model}...")
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
            'label': f'{m:<31} | GPU0: 100%, GPU1: 0%     | ~?? t/s | 128k ctx | Abliterated: ?   | Local/Ollama',
        })
        
    last_family = None
    for lm in LLAMACPP_MODELS:
        family = lm.get('family', 'Other')
        if last_family is not None and family != last_family:
            entries.append({'label': '', 'is_header': True})
        last_family = family
        entry = dict(lm)
        entries.append(entry)

    entries.append({'label': '', 'is_header': True})
    entries.append({'label': '--- API Models ---', 'is_header': True})
    vertex_models = []
    for cm in CLOUD_MODELS:
        entry = dict(cm)
        if cm.get('provider') == 'vertex':
            entry['label'] = f"Vertex AI - {cm['model']} (Billing: {cm['project_id']})"
            vertex_models.append(entry)
        else:
            entry['label'] = f"{cm['model']:<31} | Req: Internet              | ~-- t/s | -- ctx   | Unrestricted: ?  | API/Cloud"
            entries.append(entry)
    
    if vertex_models:
        entries.append({'label': '', 'is_header': True})
        entries.append({'label': '--- Vertex AI Models ---', 'is_header': True})
        entries.extend(vertex_models)
        
    return entries

def select_model(menu_entries, label):
    """Display unified model menu and return selected model config."""
    print(f'\n[MENU] {label}')
    selectable = []
    for entry in menu_entries:
        if entry.get('is_header'):
            if entry['label'] == '':
                print("")
            else:
                print(f" {entry['label']}")
        else:
            selectable.append(entry)
            print(f" {len(selectable):>2}. {entry['label']}")
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
        if entry.get('model') == model_name:
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
        
        # Shield cleanup from Ctrl+C to guarantee VRAM release
        import signal
        try:
            old_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            old_sigint = None
            
        try:
            print("[SESSION] Exiting... (Ctrl+C disabled during cleanup)")
            
            terminate_all_sub_agents()
            cleanup_transient_tools()
            
            if self._models_used:
                unregister_models_for_agent(self._models_used)
            
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
                
        finally:
            if old_sigint:
                signal.signal(signal.SIGINT, old_sigint)
            print("[SESSION] Cleanup complete.")


def terminate_all_sub_agents():
    """Find and terminate all running sub-agents using their pid.txt files."""
    print("[SYSTEM] Terminating all active sub-agents...")
    output_dir = Path("aeon_output")
    if not output_dir.exists():
        print("[SYSTEM] No sub-agents directory found. Skipping.")
        return

    terminated_count = 0
    for pid_file in output_dir.rglob("pid.txt"):
        if "sub_agents" in pid_file.parts:
            try:
                pid_str = pid_file.read_text().strip()
                if pid_str:
                    pid = int(pid_str)
                    os.kill(pid, signal.SIGKILL)
                    try:
                        os.waitpid(pid, 0)
                    except ChildProcessError:
                        pass
                    terminated_count += 1
            except (ValueError, ProcessLookupError, PermissionError):
                pass
            
            status_file = pid_file.parent / "status.txt"
            if status_file.exists():
                try:
                    status_file.write_text("KILLED")
                except:
                    pass
    if terminated_count > 0:
        print(f"[SYSTEM] Terminated {terminated_count} sub-agents.")

def _restore_backup(aeon_code_dir, backup_exists):
    """Restore the aeon source directory from the tarball backup."""
    import tarfile
    import shutil

    if not backup_exists or not os.path.exists(RESTART_BACKUP_PATH):
        print('[RESTART] No backup available to restore.')
        return False

    try:
        aeon_pkg_dir = os.path.join(aeon_code_dir, 'aeon')

        # Remove the broken code
        if os.path.isdir(aeon_pkg_dir):
            shutil.rmtree(aeon_pkg_dir)

        # Extract the backup
        with tarfile.open(RESTART_BACKUP_PATH, 'r:gz') as tar:
            tar.extractall(path=aeon_code_dir)

        # Clean up backup file
        os.remove(RESTART_BACKUP_PATH)

        print('[RESTART] Source restored from backup successfully.')
        return True
    except Exception as e:
        print(f'[RESTART] CRITICAL: Backup restoration failed: {e}')
        print(f'[RESTART] Manual recovery may be needed. Backup at: {RESTART_BACKUP_PATH}')
        return False


def _execute_restart(session, worker=None):
    """If restart_aeon was called, back up code, smoke test, reinstall, and re-exec.

    Safety sequence:
    1. Create a tarball backup of the aeon source directory
    2. Clear __pycache__ and reinstall via pip
    3. Run smoke_test.py to verify the new code is importable
    4. If smoke test fails: restore from backup, delete state file, return
       (agent continues running old code)
    5. If smoke test passes: os.execv to relaunch with --resume

    On success, this function never returns (os.execv replaces the process).
    On failure, it restores the backup, injects an error observation into the
    worker, and returns the objective string so the caller can re-run worker.run().
    Returns None if no restart was pending.
    """
    if not os.path.exists(RESTART_STATE_PATH):
        return

    terminate_all_sub_agents()

    import shutil
    import tarfile

    aeon_code_dir = None
    backup_created = False

    try:
        with open(RESTART_STATE_PATH, 'r', encoding='utf-8') as f:
            state = json.load(f)

        objective = state.get('objective', '')
        aeon_code_dir = state.get('aeon_code_dir')
        if not aeon_code_dir or not os.path.isdir(aeon_code_dir):
            print(f'[RESTART] ERROR: Invalid aeon_code_dir: {aeon_code_dir}')
            os.remove(RESTART_STATE_PATH)
            if worker:
                worker.last_observation = f'RESTART FAILED: Invalid aeon_code_dir: {aeon_code_dir}. Fix the path and try again.'
                worker.action_log.append(f'[RESTART FAILED] Invalid aeon_code_dir: {aeon_code_dir}')
                return objective
            return None

        aeon_pkg_dir = os.path.join(aeon_code_dir, 'aeon')
        if not os.path.isdir(aeon_pkg_dir):
            print(f'[RESTART] ERROR: No aeon/ package directory found in {aeon_code_dir}')
            os.remove(RESTART_STATE_PATH)
            if worker:
                worker.last_observation = f'RESTART FAILED: No aeon/ package found in {aeon_code_dir}. Fix the directory structure and try again.'
                worker.action_log.append(f'[RESTART FAILED] No aeon/ package in {aeon_code_dir}')
                return objective
            return None

        # Phase 1: Backup the aeon source directory
        print(f'[RESTART] Creating backup of aeon source...')
        try:
            if os.path.exists(RESTART_BACKUP_PATH):
                os.remove(RESTART_BACKUP_PATH)
            with tarfile.open(RESTART_BACKUP_PATH, 'w:gz') as tar:
                tar.add(aeon_pkg_dir, arcname='aeon')
            backup_created = True
            backup_size = os.path.getsize(RESTART_BACKUP_PATH)
            print(f'[RESTART] Backup created ({backup_size / 1024:.0f} KB).')
        except Exception as e:
            print(f'[RESTART] WARNING: Backup failed: {e}. Proceeding without safety net.')

        # Phase 2: Clear bytecode caches
        print(f'[RESTART] Clearing __pycache__ directories...')
        pycache_count = 0
        for root, dirs, files in os.walk(aeon_code_dir):
            for d in dirs:
                if d == '__pycache__':
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                    pycache_count += 1
        print(f'[RESTART] Cleared {pycache_count} __pycache__ directories.')

        # Phase 3: Reinstall
        print(f'[RESTART] Reinstalling aeon from {aeon_code_dir}...')
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', '.', '--quiet'],
            cwd=aeon_code_dir,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f'[RESTART] ERROR: pip install failed:\n{result.stderr}')
            _restore_backup(aeon_code_dir, backup_created)
            os.remove(RESTART_STATE_PATH)
            if worker:
                worker.last_observation = f'RESTART FAILED: pip install failed. Backup restored, old code is running.\nError: {result.stderr[:500]}\nFix the code and try restart_aeon again.'
                worker.action_log.append(f'[RESTART FAILED] pip install error. Backup restored.')
                return objective
            return None
        print('[RESTART] Reinstall complete.')

        # Phase 4: Smoke test
        smoke_test_path = os.path.join(aeon_code_dir, 'aeon', 'smoke_test.py')
        if os.path.exists(smoke_test_path):
            print('[RESTART] Running smoke test...')
            smoke_result = subprocess.run(
                [sys.executable, '-B', smoke_test_path],
                capture_output=True, text=True,
                timeout=30,
                cwd=aeon_code_dir
            )
            if smoke_result.returncode != 0:
                smoke_output = (smoke_result.stdout or '') + (smoke_result.stderr or '')
                print(f'[RESTART] SMOKE TEST FAILED. Output:')
                if smoke_result.stdout:
                    print(smoke_result.stdout)
                if smoke_result.stderr:
                    print(smoke_result.stderr)
                print('[RESTART] Restoring backup and aborting restart...')
                _restore_backup(aeon_code_dir, backup_created)
                # Reinstall the restored code
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-e', '.', '--quiet'],
                    cwd=aeon_code_dir, capture_output=True
                )
                os.remove(RESTART_STATE_PATH)
                print('[RESTART] Backup restored. Agent will continue with old code.')
                if worker:
                    worker.last_observation = (
                        f'RESTART FAILED: Smoke test detected errors in your code. '
                        f'Backup restored, old code is running.\n'
                        f'Smoke test output:\n{smoke_output[:1000]}\n'
                        f'Fix the errors above, then call restart_aeon again.'
                    )
                    worker.action_log.append(f'[RESTART FAILED] Smoke test failed. Backup restored. Errors: {smoke_output[:300]}')
                    return objective
                return None
            print('[RESTART] Smoke test passed.')
        else:
            print('[RESTART] WARNING: No smoke test found, skipping validation.')

        # Phase 5: Re-exec with --resume
        original_cwd = state.get('original_cwd', os.getcwd())
        os.chdir(original_cwd)

        new_args = [
            sys.executable, '-B', '-m', 'aeon.main',
            '--resume', RESTART_STATE_PATH,
            '--no-warmup',
        ]
        if state.get('debug_mode'):
            new_args.append('--debug')
        model_name = state.get('model_name')
        if model_name:
            new_args.extend(['--model', model_name])

        # Clean up backup on successful restart
        if backup_created and os.path.exists(RESTART_BACKUP_PATH):
            os.remove(RESTART_BACKUP_PATH)

        print(f'[RESTART] Relaunching: {" ".join(new_args)}')
        os.execv(sys.executable, new_args)
        # os.execv never returns on success

    except Exception as e:
        print(f'[RESTART] ERROR during restart: {e}')
        import traceback
        traceback.print_exc()
        _restore_backup(aeon_code_dir, backup_created)
        if os.path.exists(RESTART_STATE_PATH):
            os.remove(RESTART_STATE_PATH)
        if worker:
            worker.last_observation = (
                f'RESTART FAILED: Unexpected error: {e}. '
                f'Backup restored (if available), old code is running.\n'
                f'Fix the issue and try restart_aeon again.'
            )
            worker.action_log.append(f'[RESTART FAILED] Exception: {e}. Backup restored.')
            return objective
        return None


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable detailed LLM call logging to ~/')
    parser.add_argument('--strong', type=str, dest='model', help='Model name - skips menu')
    parser.add_argument('--model', type=str, help='Model name - skips menu (alias for --strong)')
    parser.add_argument('--weak', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--start', type=str, help='Initial objective to start immediately')
    parser.add_argument('--no-warmup', action='store_true', help='Skip model warmup (faster startup, slower first query)')
    parser.add_argument('--resume', type=str, default=None, help='Path to restart state file (used internally by restart_aeon)')
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

    try:
        session.enter(strong_config=strong_config, weak_config=weak_config, skip_warmup=args.no_warmup)
        llm_client = LLMClient(strong_config=strong_config, weak_config=weak_config)
        worker = Worker(llm_client=llm_client, debug_mode=args.debug)
        worker.model_name = strong_config['model']
        worker.model_config = strong_config
        deps = {'llm_client': llm_client, 'worker': worker}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        worker.register_tools(tools)

        prov = strong_config['provider'].upper()
        print(f"\n\033[93mAeon Ready (Model: {strong_config['model']} [{prov}], Debug: {args.debug})\033[0m")

        # --- Resume from restart if applicable ---
        if args.resume and os.path.exists(args.resume):
            try:
                with open(args.resume, 'r', encoding='utf-8') as f:
                    resume_state = json.load(f)
                os.remove(args.resume)
                worker.restore_state(resume_state)
                obj = resume_state.get('objective', '')
                print(f"[RESUME] State restored. Continuing objective: {obj}")
                while obj:
                    worker.run(obj)
                    obj = _execute_restart(session, worker)
            except Exception as e:
                print(f"[RESUME] Failed to restore state: {e}. Starting fresh.")
                import traceback
                traceback.print_exc()
                if os.path.exists(args.resume):
                    os.remove(args.resume)

        if args.start:
            obj = args.start
            while obj:
                worker.run(obj)
                obj = _execute_restart(session, worker)

        while True:
            try:
                obj = input("> ")
                if obj.strip(): 
                    if obj.strip() in ['exit', 'quit']: break
                    while obj:
                        worker.run(obj)
                        obj = _execute_restart(session, worker)
            except (KeyboardInterrupt, EOFError):
                print("\n")
                break
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        raise
    finally:
        session.exit()

if __name__ == "__main__": cli()
