import os, argparse, json, time, sys, subprocess, requests, fcntl, signal, atexit
from pathlib import Path

# Constants for locks and registry
LOCK_FILE_PATH = "/tmp/aeon_runtime.lock"
RESTART_STATE_PATH = f"/tmp/aeon_restart_state_{os.getpid()}.json"
RESTART_BACKUP_PATH = f"/tmp/aeon_restart_backup_{os.getpid()}.tar.gz"
STARTUP_LOCK_PATH = "/tmp/aeon_brain_startup.lock"
MODEL_REGISTRY_PATH = "/tmp/aeon_model_registry.json"
MODEL_REGISTRY_LOCK_PATH = "/tmp/aeon_model_registry.lock"

# Aeon is LOCAL-ONLY: no cloud/API model definitions exist by design, so nothing
# can leak prompts or context out to the web.

# =============================================================================
# LLAMA.CPP SERVED MODELS
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
        'label': 'Gemma-4-31B Native MTP Cluster  | Symmetrical Dual 256k    | ~100+ t/s | 256k ctx | Abliterated: Yes | Local/llama.cpp',
        'provider': 'llamacpp',
        'base_url': 'http://localhost:8013/v1',
        'context_limit': 262144,
        'container_name': 'aeon_gemma_mtp_lb',
        'additional_containers': ['aeon_gemma4_mtp_node0', 'aeon_gemma4_mtp_node1'],
        'start_script': 'start_gemma4_mtp.sh',
        'health_port': 8013,
    },
    # Gemma-4-31B-NVFP4 (vLLM) removed: its source LilaRest/gemma-4-31B-it-NVFP4-turbo
    # is the STOCK/censored model and was mislabeled "Abliterated: Yes". The abliterated
    # Q8_0 MTP entry above is the canonical uncensored Gemma path.
]

def is_container_running(name):
    try: return bool(subprocess.check_output(["docker", "ps", "-q", "-f", f"name={name}"], stderr=subprocess.DEVNULL, text=True).strip())
    except: return False

def wait_for_service(name, port, endpoint="/api/tags", timeout=60):
    print(f"Waiting for {name} (Port {port})...", end='', flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"http://localhost:{port}{endpoint}", timeout=2).status_code == 200: 
                print(" OK.")
                return True
        except: pass
        time.sleep(2)
        print(".", end='', flush=True)
    print(" Timeout!")
    return False

def start_local_brain_services():
    if is_container_running("aeon_brain_node"):
        print("[SYSTEM] Brain node already running.")
        return True
    print("\n[SYSTEM] Booting Local Brain...")
    script = Path(__file__).parent / "scripts" / "start_brain.sh"
    env = os.environ.copy()
    env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    subprocess.run(["bash", str(script)], check=True, env=env)
    return wait_for_service("Aeon Brain (Ollama)", 8000, endpoint="/api/tags", timeout=120)

def warm_up_models(local_model_names):
    if not local_model_names:
        return
    print("[SYSTEM] Warming up models (preloading to VRAM)...")
    models_to_warm = list(dict.fromkeys(local_model_names))
    for model in models_to_warm:
        try:
            print(f"[SYSTEM]    >> Loading {model}...", end='', flush=True)
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
        subprocess.run("docker ps -a -q --filter 'name=aeon_research' | xargs -r docker rm -f", 
                       shell=True, stderr=subprocess.DEVNULL, timeout=5)
        my_pid = os.getpid()
        def _safe_cleanup(registry_path, lock_path, container_name, cleanup_callback=None):
            try:
                with open(lock_path, 'w') as lock_fd:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    if os.path.exists(registry_path):
                        with open(registry_path, 'r') as f:
                            data = json.load(f)
                        active_pids = data.get("pids", data) if isinstance(data, dict) else data
                        other_alive_pids = []
                        if isinstance(active_pids, list):
                            for p in active_pids:
                                if not isinstance(p, int): continue
                                if p == my_pid: continue
                                try:
                                    os.kill(p, 0)
                                    with open(f"/proc/{p}/cmdline", "r") as cmd_f:
                                        if "aeon" in cmd_f.read().replace('\x00', ' ').lower():
                                            other_alive_pids.append(p)
                                except (OSError, FileNotFoundError):
                                    pass
                        if cleanup_callback:
                            cleanup_callback()
                        if not other_alive_pids:
                            subprocess.run(["docker", "rm", "-f", container_name], stderr=subprocess.DEVNULL)
            except:
                pass
        _safe_cleanup("/tmp/aeon_comfyui_registry.json", "/tmp/aeon_comfyui_registry.lock", "aeon_comfyui")
        _safe_cleanup("/tmp/aeon_vision_vllm_registry.json", "/tmp/aeon_vision_vllm_registry.lock", "aeon_qwen36_vl")
        def _close_browser_session():
            try:
                requests.post("http://localhost:8030/close_session", json={"session_id": str(my_pid)}, timeout=2)
            except:
                pass
        _safe_cleanup("/tmp/aeon_browser_registry.json", "/tmp/aeon_browser_registry.lock", "aeon_browser", _close_browser_session)
    except Exception as e:
        print(f"[WARN] Cleanup timed out or failed: {e}")

def is_llamacpp_model(config):
    return config and config.get('provider') in ['llamacpp', 'vllm']

def get_llamacpp_config(model_name):
    for m in LLAMACPP_MODELS:
        if m['model'] == model_name:
            return m
    return None

def start_llamacpp_server(config):
    container_name = config['container_name']
    port = config['health_port']
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
    print(f"[LLAMACPP] Waiting for {config['model']} to initialize. This can take 5-10 minutes if compiling kernels...")
    return wait_for_service(config['model'], port, endpoint="/health", timeout=900)

def stop_llamacpp_server(config):
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

def _cleanup_stale_pids(registry):
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
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat_content = f.read().split()
                if len(stat_content) > 2 and stat_content[2] == 'Z':
                    return False
        except FileNotFoundError:
            return False
        try:
            with open(f"/proc/{pid}/cmdline", "r") as f:
                cmdline = f.read().replace('\x00', ' ').strip().lower()
                if "aeon.main" not in cmdline and "sub_agent_wrapper" not in cmdline and not cmdline.endswith("aeon"):
                    return False
        except FileNotFoundError:
            return False
        return True
    except OSError:
        return False

def cleanup_ghost_llamacpp_containers():
    print("[SYSTEM] Scanning for ghost llama.cpp containers...")
    try:
        res = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], 
            capture_output=True, text=True, check=True
        )
        running_containers = res.stdout.splitlines()
        registry = {}
        if os.path.exists(MODEL_REGISTRY_PATH):
            try:
                with open(MODEL_REGISTRY_PATH, 'r') as f:
                    registry = json.load(f)
            except: pass
        ghosts_killed = 0
        for container in running_containers:
            if not container.startswith("aeon_"):
                continue
            matching_config = next((c for c in LLAMACPP_MODELS if c['container_name'] == container), None)
            if not matching_config:
                continue
            model_name = matching_config['model']
            pids = registry.get(model_name, [])
            if not pids or not any(_pid_exists(p) for p in pids):
                print(f"[SYSTEM] Found ghost container {container} (Model: {model_name}). Terminating...")
                subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ghosts_killed += 1
        if ghosts_killed:
            print(f"[SYSTEM] Cleaned up {ghosts_killed} ghost llama.cpp container(s).")
    except Exception as e:
        print(f"[WARN] Ghost cleanup failed: {e}")

def register_models_for_agent(models):
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

def terminate_all_sub_agents():
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
                        for _ in range(10):
                            try:
                                os.kill(pid, 0)
                                time.sleep(0.1)
                            except OSError:
                                break
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

class SessionManager:
    def __init__(self):
        self.runtime_lock = None
        self.startup_lock = None
        self._cleanup_done = False
        self._original_sigint = None
        self._original_sigterm = None
        self._models_used = []
        self._llamacpp_configs = []

    def enter(self, strong_config=None, weak_config=None, skip_warmup=False):
        local_models = []
        if strong_config and strong_config.get('provider') == 'local':
            local_models.append(strong_config['model'])
        if weak_config and weak_config.get('provider') == 'local':
            local_models.append(weak_config['model'])
        local_models = list(dict.fromkeys(local_models))
        self._models_used = list(local_models)
        llamacpp_configs = []
        for cfg in [strong_config, weak_config]:
            if cfg and is_llamacpp_model(cfg):
                if cfg not in llamacpp_configs:
                    llamacpp_configs.append(cfg)
        self._llamacpp_configs = llamacpp_configs
        needs_brain = len(local_models) > 0
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
        for lcfg in llamacpp_configs:
            model_name = lcfg['model']
            register_models_for_agent([model_name])
            self._models_used.append(model_name)
            if not start_llamacpp_server(lcfg):
                print(f"[SESSION] WARNING: Failed to start llama.cpp server for {model_name}")
        if local_models:
            register_models_for_agent(local_models)
        self.runtime_lock = open(LOCK_FILE_PATH, 'w+')
        fcntl.flock(self.runtime_lock, fcntl.LOCK_SH)
        print("[SESSION] Acquired runtime lock (agent active).")
        self._original_sigint = None
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._atexit_handler)

    def _signal_handler(self, signum, frame):
        print(f"\n[SESSION] Received SIGTERM, cleaning up...")
        self.exit()
        sys.exit(0)

    def _atexit_handler(self):
        self.exit()

    def exit(self):
        if self._cleanup_done:
            return
        self._cleanup_done = True
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