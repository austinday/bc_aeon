# LIVE TEST RESTART 2026-05-15
import os, argparse, json, time, sys, subprocess, requests, fcntl, signal, atexit, stat, threading, tempfile
from contextlib import contextmanager

import psutil

# Loading readline patches the built-in input() with a full line editor
# (wrap-aware backspace, arrow keys, history, paste) for every prompt in the
# process — the model picker, the REPL, and the shared console reader. Without it,
# input() can't backspace across a line that wrapped to a second screen row.
try:
    import readline  # noqa: F401
except Exception:
    pass

from pathlib import Path
from aeon.core.logger import get_logger
from aeon.core.presence import Presence
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

LOCK_FILE_PATH = "/tmp/aeon_runtime.lock"
RESTART_STATE_PATH = f"/tmp/aeon_restart_state_{os.getpid()}.json"
RESTART_BACKUP_PATH = f"/tmp/aeon_restart_backup_{os.getpid()}.tar.gz"
STARTUP_LOCK_PATH = "/tmp/aeon_brain_startup.lock"
QWEN_RUNTIME_ROOT = Path("/home/aday/.aeon/runtime/qwen38")
QWEN_RUNTIME_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
_qwen_root_metadata = QWEN_RUNTIME_ROOT.lstat()
if (
    not stat.S_ISDIR(_qwen_root_metadata.st_mode)
    or _qwen_root_metadata.st_uid != os.geteuid()
    or _qwen_root_metadata.st_mode & 0o077
):
    raise RuntimeError("Qwen durable runtime root is not private and owned")
QWEN_STARTUP_LOCK_PATH = str(QWEN_RUNTIME_ROOT / "lifecycle.lock")
MODEL_REGISTRY_PATH = str(QWEN_RUNTIME_ROOT / "model_registry.json")
MODEL_REGISTRY_LOCK_PATH = str(QWEN_RUNTIME_ROOT / "model_registry.lock")
QWEN_LEASE_PATH = QWEN_RUNTIME_ROOT / "lease.json"
FLEET_LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
MODEL_REGISTRY_SCHEMA_VERSION = 3
_QWEN_LIFECYCLE_THREAD_LOCK = threading.RLock()
_QWEN_LIFECYCLE_LOCAL = threading.local()


class _RetryQwenAdmission(RuntimeError):
    """Exact lost startup was reconciled; return to bounded admission wait."""


class _RetryExactQwenClaim(RuntimeError):
    """Retry verification/readiness of the same exact still-held claim."""


@contextmanager
def _qwen_lifecycle_lock():
    """Serialize every Qwen owner/registry/start/stop decision cross-process."""

    with _QWEN_LIFECYCLE_THREAD_LOCK:
        depth = int(getattr(_QWEN_LIFECYCLE_LOCAL, "depth", 0))
        if depth:
            _QWEN_LIFECYCLE_LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _QWEN_LIFECYCLE_LOCAL.depth = depth
            return
        descriptor = os.open(
            QWEN_STARTUP_LOCK_PATH,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
        ):
            os.close(descriptor)
            raise RuntimeError("Qwen lifecycle lock is not an owned regular file")
        os.fchmod(descriptor, 0o600)
        lock_file = os.fdopen(descriptor, "r+")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            _QWEN_LIFECYCLE_LOCAL.depth = 1
            yield
        finally:
            _QWEN_LIFECYCLE_LOCAL.depth = 0
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

# Aeon's control model is LOCAL-ONLY. There are deliberately no cloud/API model
# definitions and no remote fallback. An independently configured external expert
# may return bounded advice after a measured stall, but it never drives the loop.

# =============================================================================
# LOCAL MODEL CATALOG -> per-machine adaptive deploy configs
# =============================================================================
# The local model list is no longer hardcoded: it is derived from the shared
# catalog (aeon.core.model_catalog) by planning each model against THIS machine's
# detected GPUs (aeon.core.deploy_planner). This makes the same repo portable
# across the 48 GB (RTX 5000) and 96 GB (RTX 6000) Blackwell machines -- each
# model is auto-deployed on coordinator-approved physical devices, always
# keeping >=64k context and MTP where a draft head exists.
from aeon.core.gpu import GpuInfo, detect_gpus, min_total_vram_gib
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
    if entry.local_model_dir:
        d = Path(aeon_home) / 'models' / entry.local_model_dir
        required = (d / 'config.json', d / 'model.safetensors.index.json',
                    d / 'BUILD_MANIFEST.json', d / 'SHA256SUMS')
        if not d.is_dir() or not all(path.is_file() for path in required):
            return False
        try:
            manifest = json.loads((d / 'BUILD_MANIFEST.json').read_text())
            return manifest.get('complete') is True and manifest.get('status') == 'validated'
        except (OSError, ValueError):
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
# ENABLED_MODEL_NAMES is an allowlist: only these catalog entries reach the menu
# (None = every deployable model). Qwen3.8 replaces all older Qwen catalog
# entries. Qwen3.8 is deliberately the only selectable language model.
ENABLED_MODEL_NAMES = {
    _catalog.QWEN38_MODEL_NAME,          # solo default + dual-GPU option
}
# Offer the dual-copy (one model per GPU) + load-balancer deployment for models
# that fit a single GPU. Of the enabled set only Qwen3.8 fits solo, so this adds
# exactly the "Qwen3.8-27B-ARA-NVFP4-MTP
# [dual-GPU]" option — two copies fronted by adaptive_lb.py so the principal and
# its sub-agents are served concurrently across both GPUs.
# One runtime claim maps to one exact GPU UUID. Multi-GPU placements stay off
# until the coordinator lease API can bind one claim per node atomically.
OFFER_DUAL_GPU = False


def build_local_model_configs():
    """Plan each ENABLED catalog model for this machine's GPUs -> menu/runtime configs.

    For models that fit a single GPU we can offer two deployment choices:
      - SOLO: one model copy under an exclusive lease. Tools require another
        coordinator-safe physical GPU or wait.
      - DUAL: two copies (one per GPU) + router for max throughput, using BOTH GPUs
        (so GPU1 is unavailable for tools). Gated off via OFFER_DUAL_GPU right now.
    Bigger models (force_split / don't fit one GPU) get a single auto plan.

    Disabled entries (see ENABLED_MODEL_NAMES) are skipped entirely.
    """
    observed_local = [
        gpu for gpu in detect_gpus()
        if gpu.total_gib >= 90.0
    ]
    # Menu construction sizes the one supported release against its measured
    # 96-GB class even when .177 is temporarily full. This is a planning profile,
    # never an availability or numeric-device fallback: the central coordinator
    # later selects an approved host and its UUID replaces index 0 before launch.
    gpus = observed_local[:1] or [
        GpuInfo(
            index=0,
            name="coordinator-selected 96GB Blackwell class",
            total_gib=97887 / 1024.0,
            free_gib=0.0,
        )
    ]
    n = len(gpus)
    configs = []
    for entry in _catalog.CATALOG:
        if ENABLED_MODEL_NAMES is not None and entry.name not in ENABLED_MODEL_NAMES:
            continue  # disabled from selection (still catalogued / on disk)
        if not _local_model_available(entry, gpus):
            continue
        solo = _plan_deploy(entry, gpus, mode='solo')
        if solo.tier == 'solo':
            # Primary: one copy with topology-aware tool placement.
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

# The abliterated Qwen3.8-27B (ARA NVFP4 + native in-checkpoint MTP, solo) is
# Aeon's main model: the picker's Enter-default on an interactive start,
# and the straight-boot model for headless (-n) / no-TTY runs. A bare Enter boots
# the preferred `.177` release, spilling only to an enabled worker capability.
# Tools obtain their own coordinator claim; `.177` physical GPU 1 is never used.
DEFAULT_MODEL = _catalog.QWEN38_MODEL_NAME

def is_container_running(name):
    try: return bool(subprocess.check_output(["docker", "ps", "-q", "-f", f"name={name}"], stderr=subprocess.DEVNULL, text=True).strip())
    except: return False


def _owned_container_pid(config):
    """Return the exact local or worker PID for the active Qwen receipt."""

    from aeon.core.qwen_fleet_runtime import remote_container_pid, remote_state
    from aeon.core.qwen_runtime import (
        current_runtime_state,
        local_container_pid,
    )

    local_state = current_runtime_state()
    worker_state = remote_state()
    if local_state is not None and worker_state is not None:
        raise RuntimeError("local and worker Qwen receipts coexist")
    if worker_state is not None:
        return remote_container_pid()
    if local_state is None:
        return None
    pid = local_container_pid()
    if pid is None and local_state.get("phase") != "preflight":
        # PID-less heartbeats are valid only while staging, before the launcher
        # transition is durably recorded. Once launch begins, an absent,
        # unreachable, or identity-mismatched container is a surfaced heartbeat
        # failure; it must never fall back to refreshing an unbound claim.
        raise RuntimeError(
            "Qwen launch has begun but no exact container PID can be verified"
        )
    return pid


def _qwen_runtime_receipts():
    """Return the mutually exclusive local/worker durable runtime receipts."""

    from aeon.core.qwen_fleet_runtime import remote_state
    from aeon.core.qwen_runtime import current_runtime_state

    local_state = current_runtime_state()
    worker_state = remote_state()
    if local_state is not None and worker_state is not None:
        raise RuntimeError("local and worker Qwen receipts coexist")
    return local_state, worker_state

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
    """Retired compatibility hook; alternate Ollama models stay disabled."""
    print("[ERROR] The Ollama brain is retired; Aeon uses Qwen3.8 on vLLM only.")
    return False

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
    utility model. The previous separate utility "brain" was removed: the strong model is
    capable and already loaded. Media generation gets its own hard-capped tool lease
    on a different coordinator-safe GPU; exclusive Qwen is never a co-location target.
    LLMClient already
    falls back to the strong model when AEON_UTILITY_* are unset,
    so we simply make sure they are unset (also clears anything inherited from a stale env).
    """
    os.environ.pop("AEON_UTILITY_BASE_URL", None)
    os.environ.pop("AEON_UTILITY_MODEL", None)
    print("[CONFIG] Support tasks run on the strong model (no separate utility model).")

def cleanup_transient_tools():
    print("[SYSTEM] Cleaning up transient tool containers...")
    try:
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
                            subprocess.run(
                                ["docker", "stop", "--time", "30", container_name],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=45,
                            )
                            subprocess.run(
                                ["docker", "rm", container_name],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=30,
                            )
            except:
                pass
        
        _safe_cleanup("/tmp/aeon_comfyui_registry.json", "/tmp/aeon_comfyui_registry.lock", "aeon_comfyui")

        def _close_browser_session():
            try:
                from aeon.tools.browser import browser_auth_headers
                requests.post(
                    "http://localhost:8030/close_session",
                    json={"session_id": str(my_pid)},
                    headers=browser_auth_headers(), timeout=2,
                )
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


def _reconcile_failed_qwen_start(lease):
    """Reconcile one exact lost startup without ever duplicating its claim."""

    try:
        from aeon.core.gpu_queue import release_vram
        from aeon.core.qwen_runtime import (
            current_runtime_state,
            reconcile_gone_qwen_runtime,
        )

        if current_runtime_state() is None:
            # Before a runtime receipt exists, coordinator release is the
            # authoritative proof that this exact claim has no tagged process.
            release_vram(
                "Aeon reconciled Qwen admission before container launch",
                QWEN_LEASE_PATH,
                expected_claim_id=str(lease["claim_id"]),
            )
            return "cleared"
        return reconcile_gone_qwen_runtime()
    except Exception:
        return "ambiguous"

def start_llamacpp_server(config):
    """Start or reuse one `.177`-preferred release-bound fleet Qwen runtime."""

    from aeon.core.compute_profile import QWEN38_VLLM_PROFILE
    from aeon.core.qwen_capabilities import (
        enabled_qwen_runtime_capabilities,
        qwen_runtime_capability,
    )
    from aeon.core.gpu_queue import (
        PeriodicLeaseHeartbeat,
        current_lease,
        heartbeat_vram,
        reserve_named_lease,
    )
    from aeon.core.qwen_fleet_runtime import (
        capability_deploy_environment,
        remote_preflight,
        remote_runtime_liveness,
        remote_state,
        reuse_managed_remote_runtime,
        stage_remote_source,
        start_managed_remote_runtime,
        stop_managed_remote_runtime,
    )
    from aeon.core.qwen_runtime import (
        QwenLeaseLostError,
        QwenRuntimeError,
        QwenRuntimeLoadingError,
        current_runtime_state,
        finalize_releasing_qwen_runtime,
        load_artifact_identity,
        local_container_pid,
        local_image_id,
        local_image_size,
        qwen_runtime_liveness,
        reconcile_gone_qwen_runtime,
        reuse_qwen_runtime,
        start_local_runtime,
        stop_qwen_runtime,
        verify_coordinator_lease,
    )

    def startup_failure_detail(exc):
        # QwenRuntimeError messages are fixed, local lifecycle diagnostics. Do
        # not echo arbitrary third-party/subprocess exceptions, which may carry
        # unreviewed output or credentials.
        if isinstance(exc, QwenRuntimeError):
            return f"{type(exc).__name__}: {exc}"
        return type(exc).__name__

    package_root = Path(__file__).resolve().parent.parent
    container_name = config["container_name"]
    port = int(config["health_port"])
    environment = os.environ.copy()
    environment["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    environment.update({str(key): str(value) for key, value in config.get("_deploy_env", {}).items()})
    model_dir = (
        Path(environment["AEON_HOME"])
        / "models"
        / environment["AEON_LOCAL_MODEL_DIR"]
    )

    state = current_runtime_state()
    worker_state = remote_state()
    if state is not None and worker_state is not None:
        raise _RetryExactQwenClaim(
            "Local and worker Qwen lifecycle receipts coexist; preserving both"
        )
    if worker_state is not None:
        capability, _current_manifest = qwen_runtime_capability(
            str(worker_state["runtime_capability_key"]), require_enabled=False
        )
        saved_manifest = str(worker_state["runtime_capability_manifest_sha256"])
        saved_source = str(worker_state["source_manifest_sha256"])
        lease = current_lease(QWEN_LEASE_PATH)
        if lease is None:
            if remote_runtime_liveness() in {"active", "exited", "gone"}:
                try:
                    if stop_managed_remote_runtime(
                        capability,
                        saved_manifest,
                        saved_source,
                        release_reason=(
                            "Aeon stopped exact worker Qwen runtime after claim loss"
                        ),
                    ):
                        raise _RetryQwenAdmission(
                            "Exact worker runtime was retired after claim loss"
                        )
                except _RetryQwenAdmission:
                    raise
                except Exception as exc:
                    raise _RetryExactQwenClaim(
                        "Worker claim was lost but exact teardown is not yet verified"
                    ) from exc
            raise _RetryExactQwenClaim(
                "Worker runtime has no local lease receipt and remains quarantined"
            )
        try:
            current_source = stage_remote_source(capability, package_root)
            pid = reuse_managed_remote_runtime(
                capability,
                _current_manifest,
                current_source,
                lease,
                container_name=container_name,
                port=port,
            )
            if pid is None:
                raise QwenRuntimeError("saved worker runtime is gone")
            heartbeat_vram(
                pid,
                f"Aeon reused verified {config['model']} worker vLLM runtime",
                QWEN_LEASE_PATH,
            )
            return True
        except QwenLeaseLostError as exc:
            if remote_runtime_liveness() in {"active", "exited", "gone"}:
                if stop_managed_remote_runtime(
                    capability,
                    saved_manifest,
                    saved_source,
                    release_reason=(
                        "Aeon stopped exact worker Qwen runtime after claim loss"
                    ),
                ):
                    raise _RetryQwenAdmission from exc
            raise _RetryExactQwenClaim(
                "Worker claim was lost but exact teardown is not yet verified"
            ) from exc
        except QwenRuntimeError as exc:
            liveness = remote_runtime_liveness()
            if liveness in {"gone", "exited"}:
                try:
                    if stop_managed_remote_runtime(
                        capability,
                        saved_manifest,
                        saved_source,
                        release_reason="Aeon reconciled an ended worker Qwen runtime",
                    ):
                        raise _RetryQwenAdmission from exc
                except _RetryQwenAdmission:
                    raise
                except Exception:
                    pass
            raise _RetryExactQwenClaim(
                f"Saved worker runtime remains {liveness}; exact claim is preserved. "
                f"Verification failed with {startup_failure_detail(exc)}"
            ) from exc

    if state is not None:
        if state.get("teardown_only") is True:
            try:
                migrated_stopped = stop_qwen_runtime(allow_lost_lease=True)
                migrated_finalized = migrated_stopped and finalize_releasing_qwen_runtime(
                    "Aeon retired an exact schema-6 Qwen runtime before readmission"
                )
            except Exception as exc:
                raise _RetryExactQwenClaim(
                    "Schema-6 Qwen migration remains teardown-only; exact cleanup "
                    f"failed with {startup_failure_detail(exc)}"
                ) from exc
            if not migrated_finalized:
                raise _RetryExactQwenClaim(
                    "Schema-6 Qwen migration remains teardown-only; exact cleanup "
                    "is not yet verified"
                )
            raise _RetryQwenAdmission(
                "Exact schema-6 Qwen runtime was retired before fresh admission"
            )
        try:
            pid = reuse_qwen_runtime(config=config, package_root=package_root)
            if pid is None:
                reconciliation = reconcile_gone_qwen_runtime()
                if reconciliation == "cleared":
                    raise _RetryQwenAdmission
                raise _RetryExactQwenClaim("saved exact runtime is not provably gone")
            heartbeat_vram(
                pid,
                f"Aeon reused verified {config['model']} local vLLM runtime",
                QWEN_LEASE_PATH,
            )
            return True
        except QwenRuntimeLoadingError as exc:
            try:
                loading_pid = local_container_pid()
                if loading_pid is None:
                    raise QwenRuntimeError("loading runtime has no exact PID")
                loading_heartbeat = config.get("_startup_heartbeat")
                if loading_heartbeat is None:
                    loading_heartbeat = PeriodicLeaseHeartbeat(
                        state_file=QWEN_LEASE_PATH,
                        note=f"Aeon {config['model']} exact runtime is still loading",
                        pid_provider=lambda: _owned_container_pid(config),
                        interval_seconds=240,
                        require_pid=True,
                    )
                    config["_startup_heartbeat"] = loading_heartbeat
                    loading_heartbeat.start(immediate=True)
                else:
                    try:
                        if loading_heartbeat.promote_to_exact_pid() != loading_pid:
                            raise QwenRuntimeError("loading heartbeat PID changed")
                        loading_heartbeat.raise_if_failed()
                    except Exception:
                        # A PeriodicLeaseHeartbeat deliberately latches failures,
                        # and its worker thread exits on a failed beat.  Exact
                        # reuse above plus the fresh immutable-PID lookup prove
                        # that it is safe to replace only this failed heartbeat;
                        # never clear the old latch and silently retain a dead
                        # thread.  The replacement's immediate exact beat must
                        # succeed before this claim returns to bounded waiting.
                        loading_heartbeat.stop()
                        loading_heartbeat = PeriodicLeaseHeartbeat(
                            state_file=QWEN_LEASE_PATH,
                            note=(
                                f"Aeon {config['model']} exact runtime recovered "
                                "its loading heartbeat"
                            ),
                            pid_provider=lambda: _owned_container_pid(config),
                            interval_seconds=240,
                            require_pid=True,
                        )
                        config["_startup_heartbeat"] = loading_heartbeat
                        loading_heartbeat.start(immediate=True)
            except Exception as heartbeat_exc:
                raise _RetryExactQwenClaim(
                    "same exact Qwen container is loading but its PID heartbeat is ambiguous"
                ) from heartbeat_exc
            raise _RetryExactQwenClaim("same exact Qwen container is still loading") from exc
        except QwenLeaseLostError as exc:
            # An exact running container without its mandatory claim must be
            # stopped under the immutable receipt before returning to admission.
            if not stop_qwen_runtime(allow_lost_lease=True) or not finalize_releasing_qwen_runtime(
                "Aeon stopped exact Qwen runtime after claim loss"
            ):
                raise _RetryExactQwenClaim(
                    "claim was lost but exact container teardown is not yet verified"
                ) from exc
            raise _RetryQwenAdmission from exc
        except _RetryQwenAdmission:
            raise
        except QwenRuntimeError as exc:
            reconciliation = reconcile_gone_qwen_runtime()
            if reconciliation == "cleared":
                raise _RetryQwenAdmission(
                    "Saved Qwen runtime failure was reconciled after "
                    f"{startup_failure_detail(exc)}"
                ) from exc
            raise _RetryExactQwenClaim(
                f"Saved runtime remains {reconciliation}; exact claim is preserved. "
                f"Verification failed with {startup_failure_detail(exc)}"
            ) from exc

    try:
        deploy = json.loads(environment["AEON_DEPLOY_PLAN"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("Qwen deployment plan is malformed") from exc
    if (
        not isinstance(deploy, dict)
        or deploy.get("tier") != "solo"
        or len(deploy.get("nodes") or []) != 1
    ):
        raise QwenRuntimeError("Qwen requires the exact solo release plan")

    capabilities, capability_manifest_sha256 = enabled_qwen_runtime_capabilities()
    selected = None
    unavailable = []
    for capability in capabilities:
        candidate = None
        try:
            if capability.runtime_adapter == "local-docker":
                cached = config.get("_local_qwen_preflight")
                if cached is None:
                    artifact = load_artifact_identity(model_dir, verify_payload=True)
                    image = str(deploy["image"])
                    image_id = local_image_id(image)
                    image_size = local_image_size(image_id)
                    cached = (artifact, image, image_id, image_size)
                    config["_local_qwen_preflight"] = cached
                artifact, image, image_id, image_size = cached
                if image_id != capability.image_id:
                    raise QwenRuntimeError("local image differs from its capability")
                candidate = {
                    "artifact": artifact,
                    "image": image,
                    "image_id": image_id,
                    "image_size": image_size,
                    "source": None,
                }
            elif capability.runtime_adapter == "remote-docker":
                remote_cache = config.setdefault("_remote_qwen_preflights", {})
                cached = remote_cache.get(capability.key)
                if cached is None:
                    cached = remote_preflight(
                        capability, capability_manifest_sha256, package_root
                    )
                    remote_cache[capability.key] = cached
                source, remote_receipt = cached
                if (
                    remote_receipt.get("image_id") != capability.image_id
                    or remote_receipt.get("model_manifest_sha256")
                    != capability.model_manifest_sha256
                    or remote_receipt.get("model_sha256s_sha256")
                    != capability.model_sha256s_sha256
                ):
                    raise QwenRuntimeError("worker preflight differs from its capability")
                candidate = {"source": source}
            else:
                raise QwenRuntimeError("enabled Qwen adapter is unsupported")
        except Exception as exc:
            unavailable.append(
                f"{capability.host}: {startup_failure_detail(exc)}"
            )
            continue

        for gpu_id in capability.allowed_physical_gpus:
            try:
                lease = reserve_named_lease(
                    required_gb=float(capability.vram_budget_gb),
                    purpose=(
                        "Aeon Qwen3.8 on-demand text and multimodal vLLM runtime "
                        f"({capability.release_profile})"
                    ),
                    state_file=QWEN_LEASE_PATH,
                    profile=QWEN38_VLLM_PROFILE,
                    timeout=0,
                    gpu_id=gpu_id,
                    host=capability.host,
                    min_vram_gb=capability.min_physical_vram_gb,
                    run_dir_root=QWEN_RUNTIME_ROOT,
                    durable_wait=False,
                    exclusive=capability.exclusive,
                )
            except TimeoutError:
                unavailable.append(
                    f"{capability.host} GPU {gpu_id}: no coordinator-safe capacity"
                )
                continue
            verify_coordinator_lease(lease)
            selected = (capability, candidate, lease)
            break
        if selected is not None:
            break

    if selected is None:
        summary = "; ".join(unavailable) or "no enabled runtime capability"
        raise _RetryQwenAdmission(
            "No release-compatible fleet GPU is currently admissible. " + summary
        )

    runtime_capability, candidate, lease = selected
    budget_gb = float(runtime_capability.vram_budget_gb)
    if runtime_capability.runtime_adapter == "local-docker":
        node = deploy["nodes"][0]
        node["ctx"] = runtime_capability.context_tokens
        node["devices"] = lease["gpu_uuid"]
        deploy["context_limit"] = runtime_capability.context_tokens
        deploy["image"] = candidate["image_id"]
        environment["AEON_DEPLOY_PLAN"] = json.dumps(
            deploy, sort_keys=True, separators=(",", ":")
        )
        environment["AEON_GPU_MEM_UTIL"] = (
            f"{runtime_capability.gpu_memory_utilization:g}"
        )
        environment["AEON_LLM_VRAM_BUDGET_GB"] = f"{budget_gb:g}"
        environment["AEON_MAX_NUM_SEQS"] = str(runtime_capability.max_num_seqs)
        environment["AEON_MAX_NUM_BATCHED"] = str(
            runtime_capability.max_batched_tokens
        )
        environment["GPU_AGENT_CLAIM_ID"] = lease["claim_id"]
        environment["GPU_LEASE_OWNER"] = lease["owner"]
        environment["GPU_LEASE_RUN_DIR"] = lease["run_dir"]
        environment["CUDA_VISIBLE_DEVICES"] = lease["gpu_uuid"]
        environment["GPU_PLANNED_VRAM_GB"] = f"{budget_gb:g}"
        environment["GPU_RESERVE_GB"] = "6"
    else:
        environment = capability_deploy_environment(
            runtime_capability, environment, lease
        )

    startup_heartbeat = PeriodicLeaseHeartbeat(
        state_file=QWEN_LEASE_PATH,
        note=(
            f"Aeon {config['model']} {runtime_capability.host} vLLM startup is active"
        ),
        pid_provider=lambda: _owned_container_pid(config),
        interval_seconds=240,
        require_pid=False,
        promote_when_pid_available=True,
    )
    startup_heartbeat.start(immediate=True)
    # Transfer this exact heartbeat into SessionManager on success.  Keeping it
    # in the config also preserves <=15m heartbeats across same-claim retries.
    config["_startup_heartbeat"] = startup_heartbeat
    try:
        if runtime_capability.runtime_adapter == "local-docker":
            start_local_runtime(
                lease,
                environment,
                package_root=package_root,
                model_dir=model_dir,
                container_name=container_name,
                image=candidate["image"],
                port=port,
                artifact_identity=candidate["artifact"],
                image_identity=candidate["image_id"],
                image_size_bytes=candidate["image_size"],
                progress_check=startup_heartbeat.raise_if_failed,
                heartbeat_promoter=startup_heartbeat.promote_to_exact_pid,
            )
        else:
            def bind_worker_pid(pid):
                exact_pid = _owned_container_pid(config)
                if exact_pid != pid:
                    raise QwenRuntimeError("worker heartbeat PID changed")
                if startup_heartbeat.promote_to_exact_pid() != pid:
                    raise QwenRuntimeError("worker startup heartbeat was not promoted")

            start_managed_remote_runtime(
                runtime_capability,
                capability_manifest_sha256,
                candidate["source"],
                lease,
                environment,
                container_name=container_name,
                port=port,
                heartbeat_pid=bind_worker_pid,
                progress_check=startup_heartbeat.raise_if_failed,
            )
        startup_heartbeat.raise_if_failed()
        return True
    except QwenLeaseLostError as exc:
        is_remote = runtime_capability.runtime_adapter == "remote-docker"
        liveness = remote_runtime_liveness() if is_remote else qwen_runtime_liveness()
        stopped = (
            stop_managed_remote_runtime(
                runtime_capability,
                capability_manifest_sha256,
                candidate["source"],
                release_reason=(
                    "Aeon stopped exact worker Qwen startup after claim loss"
                ),
            )
            if is_remote and liveness in {"active", "gone", "exited"}
            else (
                liveness in {"active", "gone", "exited"}
                and stop_qwen_runtime(allow_lost_lease=True)
                and finalize_releasing_qwen_runtime(
                    "Aeon stopped exact Qwen startup after coordinator claim loss"
                )
            )
        )
        if stopped:
            startup_heartbeat.stop()
            config.pop("_startup_heartbeat", None)
            raise _RetryQwenAdmission from exc
        raise _RetryExactQwenClaim(
            "Qwen admission changed while exact startup state remains quarantined"
        ) from exc
    except Exception as exc:
        failure = startup_failure_detail(exc)
        is_remote = runtime_capability.runtime_adapter == "remote-docker"
        liveness = remote_runtime_liveness() if is_remote else qwen_runtime_liveness()
        if liveness == "active":
            raise _RetryExactQwenClaim(
                f"Exact Qwen container remains active after {failure}"
            ) from exc
        if is_remote:
            try:
                reconciliation = (
                    "cleared"
                    if stop_managed_remote_runtime(
                        runtime_capability,
                        capability_manifest_sha256,
                        candidate["source"],
                        release_reason="Aeon reconciled a failed worker Qwen startup",
                    )
                    else "ambiguous"
                )
            except Exception:
                reconciliation = "ambiguous"
        else:
            reconciliation = reconcile_gone_qwen_runtime()
        if reconciliation == "cleared":
            startup_heartbeat.stop()
            config.pop("_startup_heartbeat", None)
            raise _RetryQwenAdmission(
                f"Qwen startup failure was reconciled after {failure}"
            ) from exc
        raise _RetryExactQwenClaim(
            f"Qwen startup state is {reconciliation}; same claim must be retried. "
            f"Root failure was {failure}"
        ) from exc
def start_llamacpp_server_serialized(config):
    """Serialize Qwen health-check/reservation/startup across Aeon processes."""
    with _qwen_lifecycle_lock():
        retry_delay = 15.0
        while True:
            try:
                return start_llamacpp_server(config)
            except (KeyboardInterrupt, SystemExit):
                # The reserve implementation reconciles an interrupted call.
                # This outer guard also covers the tiny boundaries between its
                # durable intent writes and subprocess/sleep exception scopes.
                from aeon.core.gpu_queue import cancel_pending_reservation

                cancel_pending_reservation(QWEN_LEASE_PATH)
                raise
            except (_RetryQwenAdmission, _RetryExactQwenClaim) as retry:
                same_claim = isinstance(retry, _RetryExactQwenClaim)
                retry_detail = str(retry) or (
                    "The same exact Qwen claim requires re-verification"
                    if same_claim
                    else "The prior exact Qwen startup was safely reconciled"
                )
                print(f"[LLAMACPP] {retry_detail}", flush=True)
                try:
                    from aeon.core.compute_profile import QWEN38_VLLM_PROFILE
                    from aeon.core.gpu_queue import _update_compute_presence

                    _update_compute_presence(
                        "unavailable" if same_claim else "waiting_for_compute",
                        QWEN38_VLLM_PROFILE,
                        (
                            (
                                "The same exact Qwen claim/container is being reverified; "
                                if same_claim else
                                "Exact lost startup was reconciled; searching enabled "
                                "Qwen fleet capabilities in preference order. "
                            )
                            + retry_detail
                            + f" Retrying in {int(retry_delay)} seconds."
                        ),
                    )
                except Exception:
                    pass
                # Foreground, bounded, and Ctrl-C cancelable. Admission retry
                # holds no lease; same-claim retry keeps its exact PID heartbeat.
                time.sleep(retry_delay)
                retry_delay = min(120.0, retry_delay * 2.0)

def stop_llamacpp_server(config):
    with _qwen_lifecycle_lock():
        return _stop_llamacpp_server_locked(config)


def _stop_llamacpp_server_locked(config):
    """Keep heartbeat until exact-ID stop, then finish the release journal."""

    from aeon.core.gpu_queue import PeriodicLeaseHeartbeat, current_lease
    from aeon.core.qwen_capabilities import qwen_runtime_capability
    from aeon.core.qwen_fleet_runtime import (
        remote_container_pid,
        remote_runtime_liveness,
        remote_state,
        stop_managed_remote_runtime,
    )
    from aeon.core.qwen_runtime import (
        current_runtime_state,
        finalize_releasing_qwen_runtime,
        qwen_runtime_liveness,
        stop_qwen_runtime,
    )

    state = current_runtime_state()
    worker_state = remote_state()
    if state is not None and worker_state is not None:
        print("[WARN] Local and worker Qwen receipts coexist; preserving both.")
        return False
    if worker_state is not None:
        capability, _current_manifest = qwen_runtime_capability(
            str(worker_state["runtime_capability_key"]), require_enabled=False
        )
        heartbeat = config.pop("_startup_heartbeat", None)
        if heartbeat is None and remote_runtime_liveness() == "active":
            try:
                heartbeat = PeriodicLeaseHeartbeat(
                    state_file=QWEN_LEASE_PATH,
                    note="Aeon last-owner exact worker Qwen teardown remains active",
                    pid_provider=remote_container_pid,
                    interval_seconds=240,
                    require_pid=True,
                ).start(immediate=True)
            except Exception as exc:
                heartbeat = None
                print(
                    "[WARN] Exact worker teardown heartbeat is unavailable "
                    f"({type(exc).__name__})."
                )
        retry_delay = 15.0
        while True:
            try:
                if stop_managed_remote_runtime(
                    capability,
                    str(worker_state["runtime_capability_manifest_sha256"]),
                    str(worker_state["source_manifest_sha256"]),
                    release_reason=(
                        "Aeon Qwen3.8 exact worker vLLM stopped after final agent exited"
                    ),
                ):
                    if heartbeat is not None:
                        heartbeat.stop()
                    print(
                        "[LLAMACPP] Verified exact worker Qwen runtime stopped and released."
                    )
                    return True
            except Exception as exc:
                print(
                    "[WARN] Exact worker Qwen stop is not yet verified "
                    f"({type(exc).__name__})."
                )
            liveness = remote_runtime_liveness()
            if liveness == "active" and heartbeat is not None:
                try:
                    heartbeat.beat_once()
                except Exception as exc:
                    print(
                        "[WARN] Exact worker teardown heartbeat will retry "
                        f"({type(exc).__name__})."
                    )
            elif liveness in {"gone", "exited"} and heartbeat is not None:
                heartbeat.stop()
                heartbeat = None
            time.sleep(retry_delay)
            retry_delay = min(120.0, retry_delay * 2.0)

    if state is None:
        lease = current_lease(QWEN_LEASE_PATH)
        if lease is not None:
            # The runtime receipt is durably written before any container
            # create. Coordinator release still refuses live process evidence,
            # so this safely closes cancellation after reserve/before receipt.
            from aeon.core.gpu_queue import release_vram

            try:
                release_vram(
                    "Aeon canceled Qwen after reserve and before container create",
                    QWEN_LEASE_PATH,
                    expected_claim_id=str(lease["claim_id"]),
                )
            except Exception:
                print("[WARN] PID-less Qwen claim release is not yet verified; preserving it.")
                return False
            return current_lease(QWEN_LEASE_PATH) is None
        return True

    heartbeat = config.pop("_startup_heartbeat", None)
    if heartbeat is None and qwen_runtime_liveness() == "active":
        try:
            heartbeat = PeriodicLeaseHeartbeat(
                state_file=QWEN_LEASE_PATH,
                note="Aeon last-owner exact Qwen teardown remains active",
                pid_provider=lambda: _owned_container_pid(config),
                interval_seconds=240,
                require_pid=True,
            ).start(immediate=True)
        except Exception as exc:
            heartbeat = None
            print(f"[WARN] Exact teardown heartbeat is unavailable ({type(exc).__name__}).")

    retry_delay = 15.0
    while True:
        try:
            if stop_qwen_runtime():
                break
        except Exception as exc:
            print(f"[WARN] Exact Qwen stop is not yet verified ({type(exc).__name__}).")
        liveness = qwen_runtime_liveness()
        if liveness == "active":
            if heartbeat is None:
                print("[WARN] Active Qwen teardown has no exact heartbeat; preserving state.")
                return False
            try:
                heartbeat.beat_once()
            except Exception as exc:
                print(f"[WARN] Exact teardown heartbeat will retry ({type(exc).__name__}).")
        elif liveness in {"gone", "exited"} and heartbeat is not None:
            # Exact GPU process is gone; cleanup/release can continue without a
            # PID heartbeat, but the foreground transaction keeps retrying.
            heartbeat.stop()
            heartbeat = None
        elif liveness == "ambiguous" and heartbeat is not None:
            # Ambiguous Docker evidence is never proof that the exact process
            # ended. Keep the exact-PID heartbeat object and retry foreground.
            try:
                heartbeat.beat_once()
            except Exception as exc:
                print(f"[WARN] Ambiguous teardown heartbeat will retry ({type(exc).__name__}).")
        time.sleep(retry_delay)
        retry_delay = min(120.0, retry_delay * 2.0)

    if heartbeat is not None:
        heartbeat.stop()
    retry_delay = 15.0
    while True:
        try:
            if finalize_releasing_qwen_runtime(
                "Aeon Qwen3.8 exact local vLLM stopped after final agent exited"
            ):
                print("[LLAMACPP] Verified exact Qwen runtime stopped and released.")
                return True
        except Exception as exc:
            print(f"[WARN] Qwen release journal is pending ({type(exc).__name__}).")
        time.sleep(retry_delay)
        retry_delay = min(120.0, retry_delay * 2.0)

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


@contextmanager
def _model_registry_lock():
    descriptor = os.open(
        MODEL_REGISTRY_LOCK_PATH,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
    ):
        os.close(descriptor)
        raise RuntimeError("model registry lock is not an owned regular file")
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "r+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def _process_reference(pid=None):
    pid = int(os.getpid() if pid is None else pid)
    process = psutil.Process(pid)
    return {
        "pid": pid,
        "process_create_time": float(process.create_time()),
    }


def _legacy_process_reference(pid):
    """Migrate a live legacy PID when possible, otherwise preserve uncertainty."""

    try:
        process = psutil.Process(int(pid))
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return {"pid": int(pid), "process_create_time": -1.0}
        cmdline = process.cmdline()
        plausible = int(pid) == os.getpid() or any(
            os.path.basename(str(part)) == "aeon"
            or "aeon.main" in str(part)
            or "sub_agent_wrapper" in str(part)
            for part in cmdline
        )
        if plausible:
            return {
                "pid": int(pid),
                "process_create_time": float(process.create_time()),
            }
        # A live legacy PID with no creation-time receipt is ambiguous. Keep it
        # as an unknown live owner so PID reuse can never authorize teardown.
        return {"pid": int(pid), "process_create_time": None}
    except (TypeError, ValueError, psutil.Error, OSError):
        return {"pid": int(pid), "process_create_time": -1.0}


def _read_model_registry():
    try:
        descriptor = os.open(
            MODEL_REGISTRY_PATH,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        try:
            local_state, worker_state = _qwen_runtime_receipts()
            if local_state is not None or worker_state is not None:
                raise RuntimeError(
                    "model registry is missing while Qwen runtime evidence is active"
                )
        except ImportError:
            pass
        return {}, set()
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_size > 1024 * 1024
    ):
        os.close(descriptor)
        raise RuntimeError("model registry is not a bounded owned regular file")
    os.fchmod(descriptor, 0o600)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            raw = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("model registry is unreadable; refusing lifecycle changes") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(raw, dict):
        raise RuntimeError("model registry is not an object")
    schema = raw.get("schema_version")
    if schema == MODEL_REGISTRY_SCHEMA_VERSION:
        models = raw.get("models")
        pending_raw = raw.get("teardown_pending")
        if not isinstance(pending_raw, list) or any(
            not isinstance(model, str) or not model or len(model) > 300
            for model in pending_raw
        ):
            raise RuntimeError("model registry teardown journal is malformed")
        if len(pending_raw) != len(set(pending_raw)):
            raise RuntimeError("model registry teardown journal has duplicates")
        teardown_pending = set(pending_raw)
    elif schema == 2:
        models = raw.get("models")
        teardown_pending = set()
    else:
        models = raw
        teardown_pending = set()
    if not isinstance(models, dict):
        raise RuntimeError("model registry model map is malformed")
    normalized = {}
    for model, entries in models.items():
        if not isinstance(model, str) or not isinstance(entries, list):
            raise RuntimeError("model registry entry is malformed")
        refs = []
        for entry in entries:
            if isinstance(entry, bool):
                raise RuntimeError("model registry process identity is malformed")
            if isinstance(entry, int):
                refs.append(_legacy_process_reference(entry))
            elif isinstance(entry, dict):
                pid = entry.get("pid")
                created = entry.get("process_create_time")
                if (
                    isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                    or (
                        created is not None
                        and (isinstance(created, bool) or not isinstance(created, (int, float)))
                    )
                ):
                    raise RuntimeError("model registry process identity is malformed")
                refs.append({"pid": pid, "process_create_time": created})
            else:
                raise RuntimeError("model registry process identity is malformed")
        normalized[model] = refs
    return normalized, teardown_pending


def _write_model_registry(registry, teardown_pending=()):
    parent = Path(MODEL_REGISTRY_PATH).parent
    descriptor, temp_path = tempfile.mkstemp(
        prefix=f".{Path(MODEL_REGISTRY_PATH).name}.", suffix=".tmp", dir=str(parent)
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(
                {
                    "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
                    "models": registry,
                    "teardown_pending": sorted(set(teardown_pending)),
                },
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, MODEL_REGISTRY_PATH)
        temp_path = ""
        os.chmod(MODEL_REGISTRY_PATH, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _cleanup_stale_pids(registry):
    """Identify dead owners without erasing the last teardown receipt.

    A final owner's PID/create-time record is the durable teardown journal until
    exact model stop succeeds.  Registration may later replace proven-dead
    records, but cleanup must not create an empty-registry crash window before
    the runtime is stopped.
    """
    cleaned = {}
    orphaned = []
    for model, references in registry.items():
        alive = [reference for reference in references if _pid_exists(reference)]
        if alive:
            cleaned[model] = alive
        else:
            cleaned[model] = references
            orphaned.append(model)
            print(f"[REGISTRY] Cleaning orphaned model '{model}' (no exact live owners)")
    return cleaned, orphaned

def _pid_exists(reference):
    """Conservatively verify one PID plus immutable process creation time."""
    try:
        if isinstance(reference, int) and not isinstance(reference, bool):
            reference = _legacy_process_reference(reference)
        if not isinstance(reference, dict):
            return False
        pid = int(reference["pid"])
        expected = reference.get("process_create_time")
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        if expected is None:
            # Unknown legacy creation time is live-but-ambiguous and therefore
            # cannot authorize stopping a shared runtime.
            return True
        return expected > 0 and abs(float(process.create_time()) - float(expected)) <= 0.02
    except (KeyError, TypeError, ValueError, psutil.Error, OSError):
        return False

def cleanup_ghost_llamacpp_containers():
    """Reconcile ownerless Qwen only under the cross-process lifecycle lock."""
    print("[SYSTEM] Scanning for ghost llama.cpp containers...")
    try:
        with _qwen_lifecycle_lock():
            with _model_registry_lock():
                registry, teardown_pending = _read_model_registry()
                registry, orphaned = _cleanup_stale_pids(registry)
                teardown_pending.update(orphaned)
                local_state, worker_state = _qwen_runtime_receipts()
                if local_state is not None or worker_state is not None:
                    for config in LLAMACPP_MODELS:
                        if not registry.get(config["model"]):
                            teardown_pending.add(config["model"])
                _write_model_registry(registry, teardown_pending)
            for config in LLAMACPP_MODELS:
                model_name = config["model"]
                references = registry.get(model_name, [])
                if any(_pid_exists(reference) for reference in references):
                    continue
                if model_name not in teardown_pending:
                    continue
                print(f"[SYSTEM] Reconciling exact ownerless runtime for '{model_name}'.")
                # stop_llamacpp_server re-enters the same lifecycle lock and
                # validates full runtime/lease/container identity. No unchecked
                # rm -f fallback is permitted here.
                if not stop_llamacpp_server(config):
                    raise RuntimeError("exact ghost teardown remains pending")
                registry.pop(model_name, None)
                teardown_pending.discard(model_name)
                with _model_registry_lock():
                    _write_model_registry(registry, teardown_pending)
    except Exception as e:
        print(f"[WARN] Ghost cleanup failed: {e}")

def register_models_for_agent(models):
    """Register this exact PID incarnation for the given models."""
    if not models:
        return
    reference = _process_reference()
    with _qwen_lifecycle_lock():
        with _model_registry_lock():
            registry, teardown_pending = _read_model_registry()
            registry, orphaned = _cleanup_stale_pids(registry)
            teardown_pending.update(orphaned)
            for model in models:
                references = [
                    item for item in registry.setdefault(model, [])
                    if _pid_exists(item)
                ]
                references = [
                    item for item in references
                    if item.get("pid") != reference["pid"]
                ]
                references.append(reference)
                registry[model] = references
                teardown_pending.discard(model)
                print(f"[REGISTRY] Registered exact process {reference['pid']} for '{model}'")
            _write_model_registry(registry, teardown_pending)
        for model in tuple(sorted(teardown_pending)):
            if registry.get(model):
                continue
            lcfg = get_llamacpp_config(model)
            if lcfg:
                print(f"[SYSTEM] Stopping orphaned llama.cpp cluster for {model}...")
                if not stop_llamacpp_server(lcfg):
                    raise RuntimeError(
                        "exact Qwen teardown is pending; owner cleanup must retry"
                    )
            else:
                print(f"[SYSTEM] Unloading orphaned Ollama model {model}...")
                try:
                    requests.post("http://localhost:8000/api/generate", json={"model": model, "keep_alive": 0}, timeout=15)
                except Exception:
                    pass
            teardown_pending.discard(model)
            with _model_registry_lock():
                _write_model_registry(registry, teardown_pending)

def unregister_models_for_agent(models):
    """Unregister this exact process and stop only while lifecycle-serialized."""
    if not models:
        return
    reference = _process_reference()
    with _qwen_lifecycle_lock():
        with _model_registry_lock():
            registry, teardown_pending = _read_model_registry()
            registry, orphaned = _cleanup_stale_pids(registry)
            teardown_pending.update(orphaned)
            for model in models:
                if model not in registry:
                    continue
                before = registry[model]
                remaining = [
                    item for item in before
                    if not (
                        item.get("pid") == reference["pid"]
                        and item.get("process_create_time") == reference["process_create_time"]
                    )
                ]
                if len(remaining) == len(before):
                    continue
                print(f"[REGISTRY] Unregistered exact process {reference['pid']} from '{model}'")
                if not any(_pid_exists(item) for item in remaining):
                    # Keep the final exact PID/create-time receipt until the
                    # model runtime is durably stopped.  This makes a transient
                    # stop failure and a process crash safely retryable.
                    registry[model] = before
                    teardown_pending.add(model)
                    print(f"[REGISTRY] Model '{model}' has no users, will unload")
                else:
                    registry[model] = remaining
                    teardown_pending.discard(model)
                    print(f"[REGISTRY] Model '{model}' still has {len(registry[model])} user(s)")
            _write_model_registry(registry, teardown_pending)
        # Hold the Qwen lifecycle lock across the final no-owner decision and
        # exact stop/release so a new register cannot race this teardown.
        for model in tuple(sorted(teardown_pending)):
            references = registry.get(model, [])
            if any(
                _pid_exists(item)
                and not (
                    item.get("pid") == reference["pid"]
                    and item.get("process_create_time")
                    == reference["process_create_time"]
                )
                for item in references
            ):
                continue
            lcfg = get_llamacpp_config(model)
            if lcfg:
                print(f"[SYSTEM] Stopping llama.cpp cluster for {model}...")
                if not stop_llamacpp_server(lcfg):
                    raise RuntimeError(
                        "exact Qwen teardown is pending; owner cleanup must retry"
                    )
            else:
                print(f"[SYSTEM] Unloading Ollama model {model}...")
                try:
                    requests.post("http://localhost:8000/api/generate", json={"model": model, "keep_alive": 0}, timeout=15)
                except Exception as e:
                    print(f"[WARN] Failed to unload {model}: {e}")
            registry.pop(model, None)
            teardown_pending.discard(model)
            with _model_registry_lock():
                _write_model_registry(registry, teardown_pending)

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

def build_model_menu(local_models=None):
    """Build the Qwen3.8-only local model menu.

    Aeon's primary model is local-only. The final menu action configures a separate,
    budgeted advisory account; it is never returned as a model config. ``local_models``
    is accepted for backward compatibility but intentionally ignored.
    """
    entries = []
    entries.append({'label': '--- Local Models ---', 'is_header': True})

    last_family = None
    for lm in LLAMACPP_MODELS:
        family = lm.get('family', 'Other')
        if last_family is not None and family != last_family:
            entries.append({'label': '', 'is_header': True})
        last_family = family
        entry = dict(lm)
        entries.append(entry)

    try:
        from aeon.core.external_expert_setup import external_expert_menu_label
        entries.append({'label': '--- Optional Escalation ---', 'is_header': True})
        entries.append({
            'label': external_expert_menu_label(),
            'menu_action': 'external_expert',
        })
    except Exception as exc:
        print(f"[WARN] External expert setup is unavailable: {exc}")

    return entries

def select_model(menu_entries, label, default_model=None):
    """Display the unified model menu and return the selected model config. If
    default_model names a selectable entry, it is marked and chosen on a bare Enter
    (so the historical fast-boot into the main model is one keystroke, while every
    other deployable model — e.g. the BF16 build, dual-GPU, DeepSeek — is a number)."""
    while True:
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
                is_default = (default_model and entry.get('model') == default_model
                              and default_idx is None)
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
                    selected = selectable[default_idx - 1]
                    break
                if choice.isdigit() and 1 <= int(choice) <= len(selectable):
                    selected = selectable[int(choice)-1]
                    break
            except (KeyboardInterrupt, EOFError): sys.exit(0)
            except Exception: pass
            print('Invalid choice.')
        if selected.get('menu_action') == 'external_expert':
            from aeon.core.external_expert_setup import configure_external_expert_interactive
            configure_external_expert_interactive()
            menu_entries = build_model_menu()
            continue
        return selected

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
    def __init__(self, *, compute_backend="coordinator"):
        if compute_backend not in {"coordinator", "broker"}:
            raise ValueError("compute_backend must be coordinator or broker")
        self.compute_backend = compute_backend
        self.runtime_lock = None
        self.startup_lock = None
        self._cleanup_done = False
        self._cleanup_in_progress = False
        self._atexit_registered = False
        self._original_sigint = None
        self._original_sigterm = None
        self._models_used = []
        self._llamacpp_configs = []  # llama.cpp model configs used by this agent
        self._lease_heartbeats = []
        self._broker_service = None

    def _start_qwen_heartbeat(self, config):
        """Attach one exact-PID heartbeat to this foreground Aeon session."""

        from aeon.core.gpu_queue import PeriodicLeaseHeartbeat

        transferred = config.pop("_startup_heartbeat", None)
        if transferred is not None:
            try:
                transferred.promote_to_exact_pid()
                transferred.raise_if_failed()
            except Exception:
                transferred.stop()
            else:
                self._lease_heartbeats.append(transferred)
                return

        heartbeat = PeriodicLeaseHeartbeat(
            state_file=QWEN_LEASE_PATH,
            note=f"Aeon session owns active {config['model']} vLLM",
            pid_provider=lambda config=config: _owned_container_pid(config),
            interval_seconds=240,
            require_pid=True,
        ).start()
        self._lease_heartbeats.append(heartbeat)

    def ensure_qwen_compute(self):
        """Reconcile a latched active-claim heartbeat loss in the foreground.

        Healthy sessions return immediately. Once a heartbeat fails, the exact
        saved runtime, tunnel, lease, and coordinator identity are revalidated
        under the Qwen lifecycle lock. A verified runtime is rebound to a fresh
        exact-PID heartbeat. Only an exact runtime proven gone is released and
        returned to bounded admission. Ambiguous evidence remains quarantined
        and is rechecked with bounded backoff, never a duplicate reservation.
        """

        if self._broker_service is not None:
            self._broker_service.ensure_ready()
            return

        failed = False
        for heartbeat in tuple(self._lease_heartbeats):
            try:
                heartbeat.raise_if_failed()
            except Exception:
                failed = True
                break
        if not failed:
            return
        if len(self._llamacpp_configs) != 1:
            raise RuntimeError(
                "Qwen heartbeat failed without one exact session runtime config"
            )

        for heartbeat in self._lease_heartbeats:
            heartbeat.stop()
        self._lease_heartbeats.clear()
        config = self._llamacpp_configs[0]
        if not start_llamacpp_server_serialized(config):
            raise RuntimeError(
                "Qwen heartbeat failed and fleet re-admission did not recover"
            )
        self._start_qwen_heartbeat(config)

    def enter(self, model_config=None, skip_warmup=False):
        """Enter the session: coordinate startup, warm models, acquire locks.
        
        Only starts/warms the local brain if the selected model is an Ollama
        model. llama.cpp / vLLM models get their own container lifecycle.
        """
        # Cover the entire registration/admission/create/readiness transaction;
        # dashboard End sends SIGTERM and must never strand intent/claim/runtime.
        if self._original_sigterm is None:
            self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        if not self._atexit_registered:
            atexit.register(self._atexit_handler)
            self._atexit_registered = True

        if (
            model_config
            and is_llamacpp_model(model_config)
            and self.compute_backend == "broker"
        ):
            from aeon.core.fleet_backend import BrokerServiceSession
            from aeon.core.presence import process_instance_id

            self._broker_service = BrokerServiceSession(
                consumer=f"aeon/{process_instance_id()}"
            )
            print("[SESSION] Requesting Qwen through the Fleet Compute broker...")
            model_config["base_url"] = self._broker_service.start()
            print(f"[SESSION] Fleet broker endpoint ready: {model_config['base_url']}")

        # Determine if the selected model is local Ollama (needs brain + registry)
        local_models = []
        if model_config and model_config.get('provider') == 'local':
            local_models.append(model_config['model'])
        local_models = list(dict.fromkeys(local_models))  # deduplicate
        self._models_used = list(local_models)

        # Determine if the model is llama.cpp served
        llamacpp_configs = []
        if (
            model_config
            and is_llamacpp_model(model_config)
            and self.compute_backend == "coordinator"
        ):
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
            if not start_llamacpp_server_serialized(lcfg):
                raise RuntimeError(
                    f"Failed to start the required server for '{model_name}'. Aborting startup. "
                    f"The container's last log lines were saved to "
                    f"/tmp/aeon_{lcfg['container_name']}.crash.log (the live container is removed "
                    f"during cleanup, so 'docker logs' won't find it). A common root cause is the "
                    f"cached image lacking support for the model architecture -- rebuild with "
                    f"./setup_environment.sh."
                )
            self._start_qwen_heartbeat(lcfg)

        # --- PHASE 2: Register local Ollama models for reference counting ---
        if local_models:
            register_models_for_agent(local_models)

        # --- PHASE 3: Runtime Lock ---
        self.runtime_lock = open(LOCK_FILE_PATH, 'w+')
        fcntl.flock(self.runtime_lock, fcntl.LOCK_SH)
        print("[SESSION] Acquired runtime lock (agent active).")

        # Signal handling was installed before Phase 1 so startup is cancelable.

    def _signal_handler(self, signum, frame):
        """Unwind startup before cleanup so create/reconcile cannot race itself."""
        print("\n[SESSION] Received SIGTERM; unwinding for exact cleanup...")
        # Do not re-enter Docker/coordinator lifecycle code from inside a Python
        # signal handler while subprocess.run may still have an in-flight
        # reserve or create request. SystemExit unwinds that call first; cli's
        # finally block then invokes the idempotent journaled exit transaction.
        raise SystemExit(128 + int(signum))

    def _atexit_handler(self):
        """Fallback cleanup on normal exit."""
        self.exit()

    def exit(self):
        """Exit the session: cleanup tools, release locks, maybe unload brain / stop containers."""
        if self._cleanup_done:
            return
        if self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        cleanup_succeeded = False
        
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

            if self._broker_service is not None:
                try:
                    self._broker_service.close()
                except Exception as exc:
                    print(f"[WARN] Fleet broker ticket release failed: {exc}")
                self._broker_service = None

            # Last-owner unregister performs exact stop while these heartbeats
            # remain live. Only after unregister/teardown is safely journaled may
            # this session drop its claim refresh.
            for heartbeat in self._lease_heartbeats:
                heartbeat.stop()
            self._lease_heartbeats.clear()
            
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
            cleanup_succeeded = True
        finally:
            self._cleanup_in_progress = False
            if cleanup_succeeded:
                self._cleanup_done = True
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
        # execv preserves the PID and process creation time.  Close this run's
        # unique manifest first so it cannot be mistaken for the replacement
        # process's newly created manifest.
        if worker is not None and getattr(worker, "presence", None) is not None:
            worker.presence.mark_exit()
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


def _should_auto_adopt_tmux(args) -> bool:
    """Whether this invocation is an ordinary attachable local CLI start."""
    if args.non_interactive:
        return False
    if os.environ.get("TMUX") or os.environ.get("AEON_REMOTE_INSTANCE_ID"):
        return False
    if os.environ.get("AEON_DISABLE_AUTO_TMUX", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return False
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _auto_adopt_tmux(args) -> bool:
    """Register and replace an ordinary CLI with its managed tmux attachment.

    Returns ``False`` only when nothing was launched and the caller should keep
    starting Aeon normally. Once the inner process exists, every failure returns
    ``True`` so we never accidentally start a duplicate outside tmux.
    """
    if not _should_auto_adopt_tmux(args):
        return False

    try:
        import getpass
        import shutil
        from aeon.remote.config import RemoteConfig
        from aeon.remote.instances import InstanceManager
        from aeon.remote.store import RemoteStore

        config = RemoteConfig.from_env(validate_server=False)
        if shutil.which(config.tmux_binary) is None:
            raise RuntimeError(f"tmux is unavailable at {config.tmux_binary}")
        config.prepare_state()
        store = RemoteStore(config.database_path)
        manager = InstanceManager(store, config)
        instance = manager.adopt_local_cli(
            workspace=os.getcwd(),
            cli_args=list(sys.argv[1:]),
            objective=args.start or "",
            max_iterations=args.max_iterations,
            model=args.model,
            actor=getpass.getuser(),
        )
    except Exception as exc:
        if getattr(exc, "launched", False):
            print(
                f"[REMOTE] A managed Aeon session was launched, but setup was incomplete: {exc}. "
                "Not starting a duplicate local process.",
                file=sys.stderr,
            )
            return True
        print(
            f"[REMOTE] Could not create a managed tmux session; continuing locally: {exc}",
            file=sys.stderr,
        )
        return False

    try:
        attach_args, attach_env = manager.tmux_attach_args(instance["id"])
    except Exception as exc:
        print(
            f"[REMOTE] Aeon is running as managed instance {instance['name']}, but "
            f"automatic attachment failed: {exc}",
            file=sys.stderr,
        )
        return True

    print(
        f"[REMOTE] Adopted as managed instance {instance['name']} "
        f"({instance['id'][:12]}). Detach with Ctrl-b d; it will keep running.",
        flush=True,
    )
    try:
        os.execvpe(attach_args[0], attach_args, attach_env)
    except OSError as exc:
        print(
            f"[REMOTE] Managed instance is still running, but tmux attach failed: {exc}",
            file=sys.stderr,
        )
    return True


def cli(argv=None):
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
    parser.add_argument(
        '--browser-profile', type=str,
        default=os.environ.get('AEON_BROWSER_PROFILE', 'default'),
        help='Persistent browser login/profile name (default: AEON_BROWSER_PROFILE or default).',
    )
    args = parser.parse_args(argv)

    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error('--max-iterations must be a positive integer')

    if args.non_interactive and not (args.start or args.resume):
        parser.error('--non-interactive requires --start "<objective>" (nothing to run headless otherwise)')

    if args.dual:
        parser.error('--dual is reserved until a coordinator-safe dual-copy profile is released')

    # Help/parser exits happen above, so they remain side-effect free. An inner
    # tmux/remote process skips this branch via its environment and starts Aeon.
    if _auto_adopt_tmux(args):
        return

    from aeon.core.fleet_backend import FleetBackendError, select_compute_backend
    try:
        compute_backend, backend_reason = select_compute_backend()
    except FleetBackendError as exc:
        parser.error(str(exc))
    print(f"[CONFIG] Compute backend: {compute_backend} ({backend_reason}).")

    # Legacy lifecycle cleanup is valid only when the compatibility coordinator
    # backend owns Qwen. A broker-owned runtime must never be inspected/stopped by
    # the old per-process lifecycle.
    if compute_backend == "coordinator":
        cleanup_ghost_llamacpp_containers()

    # Register ordinary CLI starts as early as practical (after argument
    # validation, so `aeon --help` does not leave a phantom run).  Presence is
    # status-only and best-effort; a local filesystem problem must not prevent
    # Aeon itself from starting.
    try:
        presence = Presence(cwd=os.getcwd())
        if os.environ.get("AEON_REMOTE_INSTANCE_ID") and not presence.remote_instance_id:
            print(
                "[PRESENCE] Ignoring invalid AEON_REMOTE_INSTANCE_ID; using a local process UUID.",
                file=sys.stderr,
            )
    except Exception as exc:
        presence = None
        print(f"[PRESENCE] Registration unavailable: {exc}", file=sys.stderr)

    # Qwen3.8 is the complete model menu. Retired Ollama models are neither
    # enumerated nor accepted through --model, even if an old container exists.
    menu = build_model_menu()

    # --- Select model (used for both planning and utility tasks) ---
    # Qwen3.8 is the sole control model, so a bare `aeon` boots it directly.
    # --menu remains an explicit configuration surface for optional escalation.
    model_name = args.model
    if args.menu and not model_name:
        # Explicitly requested the interactive picker (choose among enabled models).
        model_config = select_model(menu, 'Select Model', default_model=DEFAULT_MODEL)
    elif model_name:
        model_config = find_model_config(model_name, menu)
        if not model_config:
            available = [e['model'] for e in menu if e.get('model')]
            print(f"[ERROR] Model '{model_name}' not found.")
            import difflib
            close = difflib.get_close_matches(model_name, available, n=3, cutoff=0.4)
            if close:
                print(f"  Did you mean: {', '.join(close)}?")
            print(f"  Available: {available}")
            sys.exit(1)
    else:
        # Interactive and headless starts both boot the one approved primary
        # model without a redundant picker.
        model_config = find_model_config(DEFAULT_MODEL, menu)
        if model_config:
            placement_label = (
                "release-bound fleet placement; 114688 ctx on .177 or 131072 ctx on .180"
                if DEFAULT_MODEL == _catalog.QWEN38_MODEL_NAME
                else model_config["label"]
            )
            print(f"[CONFIG] Booting default model: {DEFAULT_MODEL} ({placement_label}). "
                  f"Pass --menu for the picker or --model NAME to choose.")
        else:
            available = [e['model'] for e in menu if e.get('model')]
            print(f"[ERROR] Default model '{DEFAULT_MODEL}' not deployable here. "
                  f"Pass --model. Available: {available}")
            sys.exit(1)

    print(f"[CONFIG] Model: {model_config['model']} ({model_config['provider']})")
    if presence is not None:
        try:
            presence.update(phase="model_selected", model=model_config['model'])
        except Exception as exc:
            print(f"[PRESENCE] Model status update unavailable: {exc}", file=sys.stderr)

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

    session = SessionManager(compute_backend=compute_backend)

    try:
        session.enter(model_config=model_config, skip_warmup=args.no_warmup)
        enable_utility_tier_if_available(model_config)
        # Vision reuse: Qwen3.8 is multimodal and serves images
        # on its own chat endpoint, so both the browser loop and analyze_image use it
        # directly — there is no separate vision model/server. main.py exports the
        # endpoint as AEON_VISION_* (inherited by sub-agents via env). A text-only
        # primary leaves these unset and analyze_image reports vision is unavailable.
        os.environ.pop("AEON_VISION_BASE_URL", None)
        os.environ.pop("AEON_VISION_MODEL", None)
        if model_config.get('multimodal') and model_config.get('base_url'):
            vis_base = model_config['base_url']
            vis_model = model_config.get('api_model') or model_config['model']
            if vis_model != _catalog.VISION_MODEL_NAME:
                raise RuntimeError(
                    f"Refusing to configure vision through '{vis_model}'. Aeon's only "
                    f"approved vision model is '{_catalog.VISION_MODEL_NAME}'.")
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
        worker = Worker(
            llm_client=llm_client,
            debug_mode=args.debug,
            debug_log_path=args.debug_log,
            presence=presence,
        )
        worker.compute_guard = session.ensure_qwen_compute
        worker.browser_profile = args.browser_profile
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
        if presence is not None:
            try:
                presence.mark_error(e)
            except Exception:
                pass
        print(f"[ERROR] Fatal error: {e}")
        raise
    finally:
        try:
            session.exit()
        finally:
            if presence is not None:
                presence.mark_exit()

if __name__ == "__main__": cli()
