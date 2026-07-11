# LIVE TEST RESTART 2026-05-15
import os, argparse, json, time, sys, subprocess, requests, fcntl, signal, atexit

# Loading readline patches the built-in input() with a full line editor
# (wrap-aware backspace, arrow keys, history, paste) for every prompt in the
# process — the model picker, the REPL, and the shared console reader. Without it,
# input() can't backspace across a line that wrapped to a second screen row.
try:
    import readline  # noqa: F401
except Exception:
    pass

# Force local source priority to prevent site-packages resolution issues
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from pathlib import Path
from aeon.core.logger import get_logger
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

LOCK_FILE_PATH = "/tmp/aeon_runtime.lock"
RESTART_STATE_PATH = f"/tmp/aeon_restart_state_{os.getpid()}.json"
RESTART_BACKUP_PATH = f"/tmp/aeon_restart_backup_{os.getpid()}.tar.gz"
STARTUP_LOCK_PATH = "/tmp/aeon_brain_startup.lock"
MODEL_REGISTRY_PATH = "/tmp/aeon_model_registry.json"
MODEL_REGISTRY_LOCK_PATH = "/tmp/aeon_model_registry.lock"

# Aeon is LOCAL-ONLY. There are deliberately no cloud/API model definitions:
# nothing may leak prompts, context, or generated content out to the web or to
# third-party APIs. Every model runs on this machine (Ollama / llama.cpp / vLLM).
# If a model fails, the agent errors out -- it never falls back to a remote model.

# =============================================================================
# LOCAL MODEL CATALOG -> per-machine adaptive deploy configs
# =============================================================================
# The local model list is no longer hardcoded: it is derived from the shared
# catalog (aeon.core.model_catalog) by planning each model against THIS machine's
# detected GPUs (aeon.core.deploy_planner). This makes the same repo portable
# across the 48 GB (RTX 5000) and 96 GB (RTX 6000) Blackwell machines -- each
# model is auto-deployed dual-copy / GPU0-split / CPU-offload to fit, always
# keeping >=64k context and MTP where a draft head exists.
from aeon.core.gpu import detect_gpus, min_total_vram_gib
from aeon.core import model_catalog as _catalog
from aeon.core.deploy_planner import plan as _plan_deploy


def _local_model_available(entry, gpus):
    """A catalog model is offered only if deployable here AND present (or, for
    vLLM, runtime-fetchable within the machine's VRAM budget)."""
    if not gpus:
        return False
    aeon_home = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    if entry.provider == 'llamacpp':
        d = Path(aeon_home) / 'models' / (entry.model_dir or '')
        try:
            # '**/' also matches zero directories, so this finds the target GGUF
            # both directly in model_dir and in a shard subdir (e.g. Q3_K-GGUF/).
            return d.is_dir() and any(d.glob('**/' + (entry.target_glob or '*.gguf')))
        except Exception:
            return False
    # vLLM fetches weights from the HF hub at runtime; offer it if it fits.
    return _catalog.fits_download(entry, min_total_vram_gib(gpus))


def _config_from_plan(entry, p, model_name):
    return {
        'model': model_name,
        # The id to send to the inference server. vLLM enforces --served-model-name, so a
        # request for the catalog/display name ('model') 404s; it must be the served name.
        # llama.cpp ignores the field, and API providers use 'model' directly -> fall back.
        'api_model': getattr(entry, 'served_name', None) or entry.name,
        'multimodal': getattr(entry, 'multimodal', False),
        'family': entry.family,
        'label': p.label,
        'provider': entry.provider,
        'base_url': p.base_url,
        'context_limit': p.context_limit,
        'container_name': p.container_name,
        'additional_containers': [c for c in p.all_containers if c != p.container_name],
        'start_script': p.launcher,
        'health_port': p.health_port,
        '_deploy_env': p.env,
    }


# --- Model selection policy -------------------------------------------------
# The picker is OPEN again (2026-07-08, for the Qwen3.5-397B addition): every
# deployable catalog model is selectable, and a bare interactive start shows the
# menu — a bare Enter still boots DEFAULT_MODEL, so fast-boot stays one keystroke.
# To restrict selection again, set this to a set of names (the 2026-07-06
# single-model lockdown was ENABLED_MODEL_NAMES = {"Qwen3.6-27B-FP8-MTP"}).
ENABLED_MODEL_NAMES = None
OFFER_DUAL_GPU = False


def build_local_model_configs():
    """Plan each ENABLED catalog model for this machine's GPUs -> menu/runtime configs.

    For models that fit a single GPU we can offer two deployment choices:
      - SOLO: the model on GPU0 only, leaving GPU1 free for image/video/vision tools
        (the default this harness expects). Listed first / recommended.
      - DUAL: two copies (one per GPU) + router for max throughput, using BOTH GPUs
        (so GPU1 is unavailable for tools). Gated off via OFFER_DUAL_GPU right now.
    Bigger models (force_split / don't fit one GPU) get a single auto plan.

    Disabled entries (see ENABLED_MODEL_NAMES) are skipped entirely — they remain in
    the catalog / on disk but never reach the menu or runtime.
    """
    gpus = detect_gpus()
    n = len(gpus)
    configs = []
    for entry in _catalog.CATALOG:
        if ENABLED_MODEL_NAMES is not None and entry.name not in ENABLED_MODEL_NAMES:
            continue  # disabled from selection (still catalogued / on disk)
        if not _local_model_available(entry, gpus):
            continue
        solo = _plan_deploy(entry, gpus, mode='solo')
        if solo.tier == 'solo':
            # Primary: GPU0-only (keeps GPU1 for tools).
            configs.append(_config_from_plan(entry, solo, entry.name))
            # Alternative: dual-copy across both GPUs, when enabled and present.
            if OFFER_DUAL_GPU and n >= 2 and not entry.force_split:
                dual = _plan_deploy(entry, gpus, mode='dual')
                if dual.tier == 'dual':
                    configs.append(_config_from_plan(entry, dual, f"{entry.name} [dual-GPU]"))
        else:
            # Too big for one GPU: single split/offload plan.
            configs.append(_config_from_plan(entry, solo, entry.name))
    return configs


LLAMACPP_MODELS = build_local_model_configs()

# The abliterated 8-bit Qwen3.6-27B (FP8 + native in-checkpoint MTP, solo on GPU0)
# is Aeon's main model: the picker's Enter-default on an interactive start, and the
# straight-boot model for headless (-n) / no-TTY runs. Matches the catalog entry name.
DEFAULT_MODEL = "Qwen3.6-27B-FP8-MTP"

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
    """Start the Ollama brain container if not already running.

    Returns True only if the brain is up and answering; returns False (never
    raises) if it could not be started — e.g. host port 8000 is already in use —
    so callers can degrade to catalog-only models instead of crashing the whole
    harness. The brain is optional: it only backs the interactive picker's list of
    locally-pulled Ollama models (the runtime no longer uses a separate Ollama
    utility model — see enable_utility_tier_if_available).
    """
    if is_container_running("aeon_brain_node"):
        print("[SYSTEM] Brain node already running.")
        return True
    print("\n[SYSTEM] Booting Local Brain...")
    script = Path(__file__).parent / "scripts" / "start_brain.sh"
    env = os.environ.copy()
    env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    try:
        subprocess.run(["bash", str(script)], check=True, env=env)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[WARN] Local Ollama brain failed to start ({e}); "
              "the host may already use port 8000. Continuing without it.")
        return False
    return wait_for_service("Aeon Brain (Ollama)", 8000, endpoint="/api/tags", timeout=120)

def warm_up_models(local_model_names):
    """Preload local models into VRAM by making initial requests."""
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

def enable_utility_tier_if_available(model_config):
    """Support tasks (skill routing, JSON repair, summarization, log/memory compression,
    interruption analysis) run on the selected strong model -- there is no separate small
    utility model. The previous GPU1 "brain" (qwen2.5:3b) was removed: the strong model is
    capable and already loaded, and dropping it frees GPU1 entirely for image/video/vision
    tools. LLMClient already falls back to the strong model when AEON_UTILITY_* are unset,
    so we simply make sure they are unset (also clears anything inherited from a stale env).
    """
    os.environ.pop("AEON_UTILITY_BASE_URL", None)
    os.environ.pop("AEON_UTILITY_MODEL", None)
    print("[CONFIG] Support tasks run on the strong model (no separate utility model).")

def cleanup_transient_tools():
    print("[SYSTEM] Cleaning up transient tool containers...")
    try:
        # Clean standard transient tools
        subprocess.run("docker ps -a -q --filter 'name=aeon_research' | xargs -r docker rm -f", 
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
                            data = json.load(f)
                        
                        # Handle both list and dict formats (e.g., ComfyUI uses a dict)
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
    # Adaptive deploy plan (tier, GPU split, ctx, MTP) computed by deploy_planner.
    for k, v in config.get('_deploy_env', {}).items():
        env[k] = v
    result = subprocess.run(['bash', str(script)], capture_output=False, env=env)
    if result.returncode != 0:
        print(f"[LLAMACPP] ERROR: Failed to start {container_name}")
        return False
        
    print(f"[LLAMACPP] Waiting for {config['model']} to initialize. This can take 5-10 minutes if compiling kernels...")
    return wait_for_service(config['model'], port, endpoint="/health", timeout=900)

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
    """Check if PID exists AND is actually an Aeon python process, ignoring recycled PIDs."""
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
        # 1. Check for zombies
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat_content = f.read().split()
                if len(stat_content) > 2 and stat_content[2] == 'Z':
                    return False
        except FileNotFoundError:
            return False
        
        # 2. Check if the command line actually belongs to Aeon
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
    """Find and terminate llama.cpp containers whose owning agent PIDs are all dead.

    Iterates by MODEL CONFIG (not by running container) so that multi-container
    clusters are handled as a unit: when a model is a ghost, BOTH its primary
    container_name AND every entry in additional_containers are torn down. The
    previous version matched only container_name, which left orphaned cluster
    nodes (e.g. aeon_gemma_4_31b_nvfp4_mtp_node0/node1) running after an agent was killed.
    """
    print("[SYSTEM] Scanning for ghost llama.cpp containers...")
    try:
        res = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        running = set(res.stdout.splitlines())

        # We need the registry to verify if they SHOULD be running
        registry = {}
        if os.path.exists(MODEL_REGISTRY_PATH):
            try:
                with open(MODEL_REGISTRY_PATH, 'r') as f:
                    registry = json.load(f)
            except: pass

        ghosts_killed = 0
        for config in LLAMACPP_MODELS:
            # Every container this model owns (load balancer + worker nodes).
            owned = [config['container_name']] + config.get('additional_containers', [])
            running_owned = [c for c in owned if c in running]
            if not running_owned:
                continue  # none of this model's containers are up

            model_name = config['model']
            pids = registry.get(model_name, [])
            if pids and any(_pid_exists(p) for p in pids):
                continue  # a live agent still owns it; leave it alone

            # Ghost: no live owner. Tear down the WHOLE cluster, not just the LB.
            print(f"[SYSTEM] Found ghost cluster for '{model_name}': {running_owned}. Terminating...")
            for c in owned:
                subprocess.run(["docker", "rm", "-f", c],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ghosts_killed += 1

        if ghosts_killed:
            print(f"[SYSTEM] Cleaned up {ghosts_killed} ghost llama.cpp container(s).")
    except Exception as e:
        print(f"[WARN] Ghost cleanup failed: {e}")

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

# Ollama models that are infrastructure, not user-selectable primaries -- e.g. the old
# qwen2.5:3b utility/"brain" model. Hidden from the menu (matched by name prefix).
HIDDEN_OLLAMA_MODELS = ("huihui_ai/qwen2.5-abliterate",)

def build_model_menu(local_models):
    """Build the menu of available LOCAL models (Ollama + llama.cpp/vLLM).

    Aeon is local-only: there are no cloud/API entries by design, so nothing can
    leak out to the web.
    """
    local_models = [m for m in local_models
                    if not any(m.startswith(h) for h in HIDDEN_OLLAMA_MODELS)]
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

    return entries

def select_model(menu_entries, label, default_model=None):
    """Display the unified model menu and return the selected model config. If
    default_model names a selectable entry, it is marked and chosen on a bare Enter
    (so the historical fast-boot into the main model is one keystroke, while every
    other deployable model — e.g. the BF16 build, dual-GPU, DeepSeek — is a number)."""
    print(f'\n[MENU] {label}')
    selectable = []
    default_idx = None
    for entry in menu_entries:
        if entry.get('is_header'):
            if entry['label'] == '':
                print("")
            else:
                print(f" {entry['label']}")
        else:
            selectable.append(entry)
            is_default = default_model and entry.get('model') == default_model and default_idx is None
            if is_default:
                default_idx = len(selectable)
            tag = '  <- default (press Enter)' if is_default else ''
            print(f" {len(selectable):>2}. {entry['label']}{tag}")
    prompt = (f'Select Model (1-{len(selectable)}) [Enter = {default_idx}]: '
              if default_idx else f'Select Model (1-{len(selectable)}): ')
    while True:
        try:
            choice = input(prompt)
            if not choice.strip() and default_idx:
                return selectable[default_idx - 1]
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

    def enter(self, model_config=None, skip_warmup=False):
        """Enter the session: coordinate startup, warm models, acquire locks.
        
        Only starts/warms the local brain if the selected model is an Ollama
        model. llama.cpp / vLLM models get their own container lifecycle.
        """
        # Determine if the selected model is local Ollama (needs brain + registry)
        local_models = []
        if model_config and model_config.get('provider') == 'local':
            local_models.append(model_config['model'])
        local_models = list(dict.fromkeys(local_models))  # deduplicate
        self._models_used = list(local_models)

        # Determine if the model is llama.cpp served
        llamacpp_configs = []
        if model_config and is_llamacpp_model(model_config):
            llamacpp_configs.append(model_config)
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
        # A required model server that fails to come up is a HARD failure: abort
        # startup (raise) so the caller's finally-block cleans up and the process
        # exits, rather than dropping the user into a prompt with no working model.
        for lcfg in llamacpp_configs:
            model_name = lcfg['model']
            register_models_for_agent([model_name])
            self._models_used.append(model_name)
            if not start_llamacpp_server(lcfg):
                raise RuntimeError(
                    f"Failed to start the required server for '{model_name}'. Aborting startup. "
                    f"The container's last log lines were saved to "
                    f"/tmp/aeon_{lcfg['container_name']}.crash.log (the live container is removed "
                    f"during cleanup, so 'docker logs' won't find it). A common root cause is the "
                    f"cached image lacking support for the model architecture -- rebuild with "
                    f"./setup_environment.sh."
                )

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
            
            if self._original_sigterm is not None:  # SIG_DFL == 0 is falsy but valid
                signal.signal(signal.SIGTERM, self._original_sigterm)
                
        finally:
            if old_sigint is not None:
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
    ckpt_ref = ""

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
                worker.last_observation = f'RESTART FAILED: No aeon_pkg_dir in {aeon_code_dir}. Fix the directory structure and try again.'
                worker.action_log.append(f'[RESTART FAILED] No aeon_pkg_dir in {aeon_code_dir}')
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

        # Phase 1b: Durable git checkpoint (in addition to the PID-scoped tarball).
        # Unlike the tarball — which is deleted on success and protects only this one
        # transition — a git checkpoint persists as a recoverable, diffable lineage and
        # is what the boot handshake (and the revert_aeon tool) roll back to.
        try:
            from aeon.core import checkpoint as _ckpt
            ck = _ckpt.create_checkpoint(aeon_code_dir, label=state.get('reason', 'self-mod'))
            if ck.get('ok'):
                ckpt_ref = ck['tag']
                print(f"[RESTART] Git checkpoint created: {ckpt_ref}")
            else:
                print(f"[RESTART] No git checkpoint ({ck.get('reason')}); relying on tarball backup.")
        except Exception as e:
            print(f"[RESTART] WARNING: git checkpoint failed: {e}.")

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
            [sys.executable, '-m', 'pip', 'install', '.', '--quiet'],
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
                    [sys.executable, '-m', 'pip', 'install', '.', '--quiet'],
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

        # Phase 4b: Unit tests (catch logic regressions smoke test can't, e.g. a
        # broken JSON/block parser that still imports fine). Same fail-safe path:
        # restore the backup and keep the old code running on failure.
        unit_test_path = os.path.join(aeon_code_dir, 'aeon', 'tests', 'test_core.py')
        if os.path.exists(unit_test_path):
            print('[RESTART] Running unit tests...')
            try:
                unit_result = subprocess.run(
                    [sys.executable, '-B', '-m', 'aeon.tests.test_core'],
                    capture_output=True, text=True, timeout=60, cwd=aeon_code_dir
                )
            except subprocess.TimeoutExpired:
                unit_result = None
                print('[RESTART] WARNING: unit tests timed out; treating as failure.')
            if unit_result is None or unit_result.returncode != 0:
                unit_output = '' if unit_result is None else (unit_result.stdout or '') + (unit_result.stderr or '')
                print('[RESTART] UNIT TESTS FAILED. Restoring backup and aborting restart...')
                if unit_output:
                    print(unit_output[-2000:])
                _restore_backup(aeon_code_dir, backup_created)
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '.', '--quiet'],
                    cwd=aeon_code_dir, capture_output=True
                )
                os.remove(RESTART_STATE_PATH)
                print('[RESTART] Backup restored. Agent will continue with old code.')
                if worker:
                    worker.last_observation = (
                        f'RESTART FAILED: Unit tests detected a regression in your code. '
                        f'Backup restored, old code is running.\n'
                        f'Unit test output (tail):\n{unit_output[-1000:]}\n'
                        f'Fix the failing tests, then call restart_aeon again.'
                    )
                    worker.action_log.append(f'[RESTART FAILED] Unit tests failed. Backup restored. Tail: {unit_output[-300:]}')
                    return objective
                return None
            print('[RESTART] Unit tests passed.')

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

        # Boot handshake: the smoke/unit gates above ran the new code as a SUBPROCESS,
        # but execv relaunches through the (untested) --resume path. Mark the boot as
        # pending and name the checkpoint to roll back to; the relaunched process clears
        # this once it boots healthy, and any fresh start that still sees it auto-reverts.
        try:
            from aeon.core import bootguard
            bootguard.mark_pending(aeon_code_dir, ckpt_ref, reason=state.get('reason', ''))
        except Exception as e:
            print(f"[RESTART] WARNING: could not write boot marker: {e}.")

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
                f'Backup restored (if available), old code is running.'
            )
            worker.action_log.append(f'[RESTART] Exception: {e}. Backup restored.')
            return objective
        return None


def cli():
    cleanup_ghost_llamacpp_containers()
    parser = argparse.ArgumentParser(
        prog='aeon',
        description='Aeon — an autonomous, self-modifying agent harness. '
                    'Runs a single LLM in a plan/act loop with collapsible tools, '
                    'skills, sub-agents, and persistent memory.',
        epilog='Examples:\n'
               '  python3 -m aeon.main --start "Summarize the repo"\n'
               '  python3 -m aeon.main --start "Build X" --max-iterations 40\n'
               '  python3 -m aeon.main -n --start "Do X and exit"   # headless, no prompt\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--debug', action='store_true', help='Enable detailed LLM call logging to ~/')
    parser.add_argument('--debug-log', type=str, help='Path to the reasoning trace log file (JSONL)')
    parser.add_argument('--model', type=str, help='Model name - skips the menu')
    parser.add_argument('--menu', '-i', action='store_true',
                        help='Force the interactive model picker (choose solo vs dual-GPU, etc.) '
                             'even when the default model is deployable.')
    parser.add_argument('--dual', action='store_true',
                        help='Deploy the main model in DUAL-GPU mode (a copy on each GPU + routing) '
                             'instead of the default single-GPU (solo) placement.')
    parser.add_argument('--start', type=str, help='Initial objective to start immediately')
    parser.add_argument('--non-interactive', '-n', action='store_true',
                        help='Headless mode: run the --start objective (and any --resume) to '
                             'completion, then exit without dropping into the interactive "> " '
                             'prompt. Never shows the model picker. Requires --start (or --resume).')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Cap iterations per objective; the agent is forced to deliver a final '
                             'report at the limit. Default: unbounded.')
    parser.add_argument('--no-warmup', action='store_true', help='Skip model warmup (faster startup, slower first query)')
    parser.add_argument('--resume', type=str, default=None, help='Path to restart state file (used internally by restart_aeon)')
    args = parser.parse_args()

    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error('--max-iterations must be a positive integer')

    if args.non_interactive and not (args.start or args.resume):
        parser.error('--non-interactive requires --start "<objective>" (nothing to run headless otherwise)')

    # --- Enumerate local (Ollama) models — only when the picker actually needs them ---
    # The Ollama "brain" container exists solely to list locally-pulled Ollama
    # models in the interactive picker; nothing at runtime uses it anymore (support
    # tasks run on the strong model — see enable_utility_tier_if_available, and the
    # GPU1 qwen2.5:3b brain was removed). Starting it pulls a ~2 GB image on first
    # run and binds host port 8000, so a headless or default/--model boot — which
    # never shows the picker and today runs a vLLM catalog model — must NOT start
    # it. Doing so both wasted that pull/boot and, if the host already used :8000,
    # crashed the entire harness, breaking the host-coexistence goal. Start it only
    # when an Ollama model might actually be chosen, and treat failure as non-fatal.
    def _enumerate_ollama_models():
        if is_container_running("aeon_brain_node"):
            return get_ollama_models()
        print("[SYSTEM] Starting local Ollama brain to enumerate local models...")
        if start_local_brain_services():
            return get_ollama_models()
        return []

    if args.menu and not args.model:
        local_models = _enumerate_ollama_models()    # picker may offer Ollama models
    elif is_container_running("aeon_brain_node"):
        local_models = get_ollama_models()            # already running — include for free
    else:
        local_models = []

    # --- Build the model menu (always includes the vLLM / llama.cpp catalog) ---
    menu = build_model_menu(local_models)

    # An explicit --model that isn't a catalog entry may name a local Ollama model
    # we skipped enumerating; start the brain once and rebuild before erroring out.
    if args.model and not local_models and not find_model_config(args.model, menu):
        extra = _enumerate_ollama_models()
        if extra:
            menu = build_model_menu(extra)

    # --- Select model (used for both planning and utility tasks) ---
    # On a bare interactive (TTY) start the picker is shown, with Qwen3.6-27B-FP8-MTP
    # as the Enter-default. Headless (-n) and no-TTY starts never prompt: they boot
    # the default directly, so scripted `--start "<task>"` pipelines just go.
    # --model NAME skips the menu either way.
    model_name = args.model
    if not model_name and args.dual:
        # Dual-GPU is currently disabled (OFFER_DUAL_GPU=False); this name won't resolve,
        # which yields the clear "not found / available" error below rather than silently
        # falling back to solo.
        model_name = f"{DEFAULT_MODEL} [dual-GPU]"

    if args.menu and not model_name:
        # Explicitly requested the interactive picker (choose among enabled models).
        model_config = select_model(menu, 'Select Model', default_model=DEFAULT_MODEL)
    elif model_name:
        model_config = find_model_config(model_name, menu)
        if not model_config:
            available = [e['model'] for e in menu if not e.get('is_header')]
            print(f"[ERROR] Model '{model_name}' not found.")
            import difflib
            close = difflib.get_close_matches(model_name, available, n=3, cutoff=0.4)
            if close:
                print(f"  Did you mean: {', '.join(close)}?")
            print(f"  Available: {available}")
            sys.exit(1)
    elif sys.stdin.isatty() and not args.non_interactive:
        # Bare interactive start: show the picker (re-enabled 2026-07-08). A bare
        # Enter boots DEFAULT_MODEL, so the historical fast boot is one keystroke.
        model_config = select_model(menu, 'Select Model', default_model=DEFAULT_MODEL)
    else:
        # Headless (-n) or no TTY: boot straight into the default model with no
        # picker to click through, so scripted `--start "<task>"` runs just go.
        model_config = find_model_config(DEFAULT_MODEL, menu)
        if model_config:
            print(f"[CONFIG] Booting default model: {DEFAULT_MODEL} (single-GPU/solo on GPU0). "
                  f"Pass --menu for the picker or --model NAME to choose.")
        else:
            available = [e['model'] for e in menu if not e.get('is_header')]
            print(f"[ERROR] Default model '{DEFAULT_MODEL}' not deployable here. "
                  f"Pass --model. Available: {available}")
            sys.exit(1)

    print(f"[CONFIG] Model: {model_config['model']} ({model_config['provider']})")

    # Boot handshake recovery: on a FRESH start (not the --resume relaunch), a still-
    # pending marker means a previous restart booted broken code and never went healthy.
    # Roll it back to its checkpoint before doing anything else. Skipped under --resume,
    # which IS the relaunch that is expected to clear the marker once it boots.
    if not args.resume:
        try:
            from aeon.core import bootguard
            bootguard.check_and_recover()
        except Exception as e:
            print(f"[BOOTGUARD] recovery check failed: {e}")

    session = SessionManager()

    try:
        session.enter(model_config=model_config, skip_warmup=args.no_warmup)
        enable_utility_tier_if_available(model_config)
        # Vision reuse: the selected primary (Gemma-4) is multimodal and serves images
        # on its own chat endpoint, so both the browser loop and analyze_image use it
        # directly — there is no separate vision model/server. main.py exports the
        # endpoint as AEON_VISION_* (inherited by sub-agents via env). A text-only
        # primary leaves these unset and analyze_image reports vision is unavailable.
        os.environ.pop("AEON_VISION_BASE_URL", None)
        os.environ.pop("AEON_VISION_MODEL", None)
        if model_config.get('multimodal') and model_config.get('base_url'):
            vis_base = model_config['base_url']
            vis_model = model_config.get('api_model') or model_config['model']
            # HARD GATE: a model we *declare* multimodal is trusted to drive the
            # browser from screenshots. Before exporting it as the vision backend,
            # prove it can actually SEE — send a nonce image and require it read
            # back. This catches an unreachable endpoint, a text-only build that
            # rejects images, AND (the silent-killer) a quant/MTP build that
            # accepts images but confabulates. If it fails, STOP with fix-it info
            # rather than let a blind model browse. Opt out with
            # AEON_SKIP_VISION_SELFTEST=1 (unverified — prints a warning).
            if os.environ.get("AEON_SKIP_VISION_SELFTEST") == "1":
                print("\033[93m[VISION SELF-TEST] Skipped via AEON_SKIP_VISION_SELFTEST=1 "
                      "— vision is UNVERIFIED this session.\033[0m")
            else:
                from aeon.core.vision_selftest import run_vision_self_test, VisionSelfTestError
                print(f"[VISION SELF-TEST] Verifying {vis_model} can read an image...")
                try:
                    code = run_vision_self_test(vis_base, vis_model)
                    if code:
                        print(f"\033[92m[VISION SELF-TEST] PASS — model read the probe code "
                              f"'{code}'. Vision trusted for browsing.\033[0m")
                except VisionSelfTestError as ve:
                    # A vision-capability failure is NOT a broken-code regression:
                    # the code booted, imported and restored state fine. Clear the
                    # bootguard pending marker first so this abort is not
                    # misattributed to a code change and rolled back on next start.
                    try:
                        from aeon.core import bootguard
                        bootguard.mark_boot_ok()
                    except Exception:
                        pass
                    bar = "=" * 72
                    imgs = "\n".join(f"        {p}" for p in getattr(ve, "images", []))
                    print(f"\n\033[91m{bar}\n"
                          f"FATAL: VISION SELF-TEST FAILED for '{vis_model}'\n"
                          f"{bar}\n"
                          f"Why:  {ve}\n"
                          f"Fix:  {ve.hint}\n"
                          f"Endpoint: {vis_base.rstrip('/')}/chat/completions\n"
                          + (f"Probe images (inspect these — they are crisp):\n{imgs}\n" if imgs else "")
                          + f"This model is declared multimodal=True but cannot be trusted to see.\n"
                          f"Aeon is stopping so this is fixed rather than silently browsing blind.\n"
                          f"(To start anyway, text-only and UNVERIFIED, set "
                          f"AEON_SKIP_VISION_SELFTEST=1.)\n"
                          f"{bar}\033[0m")
                    raise
            os.environ["AEON_VISION_BASE_URL"] = vis_base
            os.environ["AEON_VISION_MODEL"] = vis_model
            print(f"[CONFIG] Vision -> reusing the loaded multimodal model "
                  f"({os.environ['AEON_VISION_MODEL']}); no separate vision server.")
        llm_client = LLMClient(model_config)
        worker = Worker(llm_client=llm_client, debug_mode=args.debug, debug_log_path=args.debug_log)
        worker.model_name = model_config['model']
        worker.model_config = model_config
        deps = {'llm_client': llm_client, 'worker': worker}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        
        # Manual override for skill manager tools to bypass loader issues
        try:
            from aeon.tools.skills_manager_tool import ExpandSkillsCategory, CollapseSkillsCategory
            manual_tools = [
                ExpandSkillsCategory(worker=worker, llm_client=llm_client),
                CollapseSkillsCategory(worker=worker, llm_client=llm_client)
            ]
            tools.extend(manual_tools)
            print("[SYSTEM] Manually registered skill manager tools.")
        except Exception as e:
            print(f"[SYSTEM] Failed to manually register skill tools: {e}")

        worker.register_tools(tools)

        # --- Startup Skills Summary ---
        try:
            from aeon.core.skills.manager import SkillsManager
            sm = SkillsManager()
            skills_dir = Path(sm.base_dir).resolve()
            if skills_dir.exists():
                root_skills = [f.stem for f in skills_dir.glob("*.txt") if not f.name.startswith('__')]
                skill_categories = [d.name for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
                
                if root_skills or skill_categories:
                    print("\n\033[92m[S-V-S-S-S] Loaded Skills:\033[0m", file=sys.stderr)
                    if root_skills:
                        for skill in sorted(root_skills):
                            print(f"  - {skill}", file=sys.stderr)
                    for cat in sorted(skill_categories):
                        skills = sm.get_skills_in_category(cat)
                        if skills:
                            print(f"  - {cat}/", file=sys.stderr)
                            for skill in sorted(skills):
                                print(f"    - {skill}", file=sys.stderr)
                else:
                    print(f"\n[SYSTEM] No skill protocols found in: {skills_dir}", file=sys.stderr)
            else:
                print(f"\n[SYSTEM] Skills directory not found at: {skills_dir}", file=sys.stderr)
        except Exception as e:
            print(f"\n[SYSTEM] Failed to load skills summary: {e}")
        prov = model_config['provider'].upper()

        # --- Startup Tool Summary ---
        try:
            from aeon.tools.categories import TOOL_CATEGORIES, TOP_LEVEL_TOOLS
            
            if TOOL_CATEGORIES or TOP_LEVEL_TOOLS:
                print("\n\033[94m[S-V-S-S-S] Loaded Tools:\033[0m")
                
                # 1. Print Top Level Tools
                top_level = sorted(list(TOP_LEVEL_TOOLS))
                if top_level:
                    for tool in top_level:
                        print(f"  - {tool}")
                
                # 2. Print Categorized Tools
                for cat_name, cat_data in sorted(TOOL_CATEGORIES.items()):
                    tools = cat_data.get('tools', [])
                    if tools:
                        print(f"  - {cat_name}/")
                        for tool in sorted(tools):
                            print(f"    - {tool}")
                    else:
                        print(f"  - {cat_name}/ (No tools found in category data)")
        except Exception as e:
            print(f"\n[SYSTEM] Failed to load tools summary: {e}")

        # --- Startup Visibility ---
        print(f"\n\033[93mAeon Ready (Model: {model_config['model']} [{prov}], Debug: {args.debug})\033[0m")

        # --- Resume from restart if applicable ---
        if args.resume and os.path.exists(args.resume):
            try:
                with open(args.resume, 'r', encoding='utf-8') as f:
                    resume_state = json.load(f)
                os.remove(args.resume)
                worker.restore_state(resume_state)
                # The relaunched code booted, imported, and restored state successfully —
                # clear the pending marker so it is treated as the new known-good generation.
                try:
                    from aeon.core import bootguard
                    bootguard.mark_boot_ok()
                except Exception:
                    pass
                obj = resume_state.get('objective', '')
                print(f"[RESUME] State restored. Continuing objective: {obj}")
                while obj:
                    worker.run(obj, max_iterations=args.max_iterations)
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
                worker.run(obj, max_iterations=args.max_iterations)
                obj = _execute_restart(session, worker)

        # Headless mode: the --start objective (and any --resume) is done; exit instead
        # of dropping into the interactive prompt. Lets you "start it up with a task and
        # it just goes" with no attached terminal.
        if args.non_interactive:
            print("[CONFIG] Non-interactive mode: objective complete, exiting.")
        else:
            from aeon.core.console import console
            while True:
                try:
                    obj = console().readline("> ")
                    if obj.strip():
                        if obj.strip() in ['exit', 'quit']: break
                        while obj:
                            worker.run(obj, max_iterations=args.max_iterations)
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
