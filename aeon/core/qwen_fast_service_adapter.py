"""Production service adapter for the gated single-GPU Qwen3.8 fast release.

This adapter deliberately reuses the exact worker, image, target, draft, runtime
arguments, and source closure exercised by the speed lab.  It changes only the
lifecycle: the verified container remains alive behind an owner-only loopback SSH
tunnel until Fleet Compute asks the adapter to stop it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import stat
import subprocess
import threading
import time
from typing import Any, Mapping

import requests

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    LaunchResult,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from .qwen_speed_lab_adapter import (
    DRAFT_ARTIFACTS,
    ENGINE_ARCHIVE_SHA256,
    HOST,
    HOST_CONFIGS,
    IMAGE_ID,
    IMAGE_SIZE_BYTES,
    MODEL_ARTIFACTS,
    PORT,
    REMOTE_RUN_ROOT,
    VARIANT_CONFIGS,
    WORKER_SCHEMA_VERSION,
    AeonQwenSpeedLabAdapter,
    QwenSpeedLabError,
    QwenSpeedLabTransportError,
    _canonical_sha256,
    _prompt_bundle,
    _remote_action,
    _remote_metrics,
    _source_manifest,
    _ssh_base,
)


PROFILE_ID = "aeon-qwen38-fast-180"
WINNER_VARIANT = (
    "nightly-v2-full-gdn-nvfp4-dflash2-k7-flashinfer-fp8kv-piecewise"
)
SERVED_MODEL = "Qwen3.8-27B-ARA-NVFP4-MTP"
LOCAL_PORT = 18034
LOCAL_ENDPOINT = f"http://127.0.0.1:{LOCAL_PORT}/v1"
EVIDENCE_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/artifacts/aeon-qwen38-speed-lab/"
    "fr-6dc0a055b21d426bbc02175877b578c2"
)
EVIDENCE_SHA256 = {
    "manifest": "3e1246bc4bcff1825eb6eab9b77af6cb21bac19e30140de48285e2204b7497bd",
    "preflight": "73c74c1fa435c988c13f5454519474fc1541f367824f49d81900f41c79e20943",
    "quality": "be8cf34daf40b7cbb72aff99637e0310c4d2a54f486e8923c99dc371e4ca818d",
    "result": "fce1f4f0996d5493837c48a5093c2af12dbab0ff1008fa0817761278b01b6f84",
    "speed": "bc9dce58ebbb3ea6b1a83ff7b4c6fa8775439dd88047c548bfedfd4d6b414573",
}
EVIDENCE_FILES = {
    "manifest": "MANIFEST.sha256",
    "preflight": "preflight.json",
    "quality": "quality.json",
    "result": "result.json",
    "speed": "speed.json",
}

_RUNTIME_ID_RE = re.compile(r"^fr-[a-f0-9]{32}$")
_PROCESS_IDENTITY_RE = re.compile(
    r"^aeon-qwen38-fast:(fr-[a-f0-9]{32}):([a-f0-9]{64}):([0-9]+)$"
)
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_TUNNEL_RECEIPT = "fast-service-tunnel.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o077
    ):
        raise QwenSpeedLabError("fast-service receipt directory is unsafe")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = (
            json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _private_json(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= 64 * 1024
    ):
        raise QwenSpeedLabError("fast-service tunnel receipt is unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QwenSpeedLabError("fast-service tunnel receipt is malformed") from exc
    if not isinstance(value, dict):
        raise QwenSpeedLabError("fast-service tunnel receipt is not an object")
    return value


def _process_start_ticks(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = payload.rfind(")")
    if end < 0:
        raise QwenSpeedLabError("fast-service tunnel process stat is malformed")
    return int(payload[end + 2 :].split()[19])


def _process_argv(pid: int) -> list[str]:
    metadata = Path(f"/proc/{pid}").stat()
    if metadata.st_uid != os.geteuid():
        raise QwenSpeedLabError("fast-service tunnel owner changed")
    payload = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if payload and payload[-1] == b"":
        payload.pop()
    try:
        return [item.decode("utf-8") for item in payload]
    except UnicodeDecodeError as exc:
        raise QwenSpeedLabError("fast-service tunnel argv is malformed") from exc


def _tunnel_argv() -> list[str]:
    base = _ssh_base(HOST)
    return [
        *base[:-1],
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"127.0.0.1:{LOCAL_PORT}:127.0.0.1:{PORT}",
        base[-1],
    ]


def _pid_slot_absent(pid: Any) -> bool:
    """Return true only when the exact PID number is absent from procfs."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    try:
        Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _bounded_loopback_body(response: requests.Response, maximum: int) -> bytes:
    if type(maximum) is not int or maximum <= 0:
        raise QwenSpeedLabError("fast-service response bound is invalid")
    payload = bytearray()
    try:
        advertised = response.headers.get("content-length")
        if advertised is not None:
            try:
                advertised_size = int(advertised)
            except (TypeError, ValueError) as exc:
                raise QwenSpeedLabError(
                    "fast-service response Content-Length is malformed"
                ) from exc
            if advertised_size < 0 or advertised_size > maximum:
                raise QwenSpeedLabError("fast-service response exceeded its bound")
        for chunk in response.iter_content(chunk_size=min(64 * 1024, maximum + 1)):
            payload.extend(chunk)
            if len(payload) > maximum:
                raise QwenSpeedLabError("fast-service response exceeded its bound")
    finally:
        response.close()
    return bytes(payload)


def _endpoint_ready() -> bool:
    try:
        health = requests.get(
            f"http://127.0.0.1:{LOCAL_PORT}/health",
            timeout=(2, 10),
            allow_redirects=False,
            proxies={"http": "", "https": ""},
            stream=True,
        )
        health_status = health.status_code
        _bounded_loopback_body(health, 64 * 1024)
        models = requests.get(
            f"{LOCAL_ENDPOINT}/models",
            timeout=(2, 10),
            allow_redirects=False,
            proxies={"http": "", "https": ""},
            stream=True,
        )
        models_status = models.status_code
        models_body = _bounded_loopback_body(models, 256 * 1024)
        if health_status != 200 or models_status != 200:
            return False
        value = json.loads(models_body)
        return {
            item.get("id")
            for item in value.get("data", [])
            if isinstance(item, dict)
        } == {SERVED_MODEL}
    except (requests.RequestException, QwenSpeedLabError, TypeError, ValueError):
        return False


def _tunnel_exact(receipt: Mapping[str, Any]) -> bool:
    pid = receipt.get("pid")
    start_ticks = receipt.get("start_ticks")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 1
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
    ):
        return False
    try:
        return _process_start_ticks(pid) == start_ticks and _process_argv(pid) == _tunnel_argv()
    except (FileNotFoundError, OSError, ValueError, QwenSpeedLabError):
        return False


def _start_tunnel(run_dir: Path, runtime_id: str, request_sha256: str) -> None:
    receipt_path = run_dir / _TUNNEL_RECEIPT
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _private_json(receipt_path)
        if (
            receipt.get("runtime_id") == runtime_id
            and receipt.get("request_sha256") == request_sha256
            and receipt.get("state") == "active"
            and _tunnel_exact(receipt)
            and _endpoint_ready()
        ):
            return
        raise QwenSpeedLabError("fast-service tunnel receipt already exists")
    intent = {
        "schema_version": 1,
        "runtime_id": runtime_id,
        "request_sha256": request_sha256,
        "state": "starting",
        "pid": None,
        "start_ticks": None,
        "created_at": time.time(),
    }
    # Publish the intent before touching the port/process.  Recovery never scans
    # global process state or adopts a matching SSH argv; an interrupted intent
    # remains an explicit operator-recovery condition.
    _atomic_json(receipt_path, intent)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        probe.bind(("127.0.0.1", LOCAL_PORT))
    except OSError as exc:
        raise QwenSpeedLabError("fast-service loopback port is unavailable") from exc
    finally:
        probe.close()
    process = subprocess.Popen(
        _tunnel_argv(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    try:
        receipt = {
            **intent,
            "state": "active",
            "pid": process.pid,
            "start_ticks": _process_start_ticks(process.pid),
        }
        _atomic_json(receipt_path, receipt)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise QwenSpeedLabError("fast-service tunnel exited before health")
            if not _tunnel_exact(receipt):
                raise QwenSpeedLabError("fast-service tunnel identity changed")
            if _endpoint_ready():
                return
            time.sleep(0.5)
        raise QwenSpeedLabError("fast-service tunnel did not become healthy")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Do not escalate to SIGKILL.  The exact tunnel remains receipted
                # and Fleet will fail closed for operator recovery.
                pass
        if receipt_path.is_file() and process.poll() is not None:
            failed_receipt = _private_json(receipt_path)
            _atomic_json(
                receipt_path,
                {**failed_receipt, "state": "stopped", "stopped_at": time.time()},
            )
        raise


def _stop_tunnel(run_dir: Path, runtime_id: str, request_sha256: str) -> bool:
    path = run_dir / _TUNNEL_RECEIPT
    try:
        receipt = _private_json(path)
    except FileNotFoundError:
        # No receipt means no authority to classify or stop any process that
        # might own the port.
        return False
    if (
        receipt.get("runtime_id") != runtime_id
        or receipt.get("request_sha256") != request_sha256
    ):
        return False
    if receipt.get("state") == "stopped":
        return _pid_slot_absent(receipt.get("pid"))
    if not _tunnel_exact(receipt):
        pid = receipt.get("pid")
        if _pid_slot_absent(pid):
            _atomic_json(path, {**receipt, "state": "stopped", "stopped_at": time.time()})
            return True
        return False
    pid = int(receipt["pid"])
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if _pid_slot_absent(pid):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            _atomic_json(path, {**receipt, "state": "stopped", "stopped_at": time.time()})
            return True
        if not _tunnel_exact(receipt):
            # A live PID with different start ticks/argv is ambiguous PID reuse
            # or identity drift, never successful exact-process termination.
            return False
        time.sleep(0.1)
    return False


class _ServiceHeartbeat:
    def __init__(self, context: RuntimeContext, pid: int) -> None:
        self.context = context
        self.pid = pid
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_ServiceHeartbeat":
        self.context.heartbeat(self.pid, "Qwen fast service loading exact release")
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise QwenSpeedLabError("fast-service startup heartbeat failed") from self.error

    def check(self) -> None:
        if self.error is not None:
            raise QwenSpeedLabError("fast-service startup heartbeat failed") from self.error

    def _run(self) -> None:
        while not self.stop.wait(240):
            try:
                self.context.heartbeat(
                    self.pid, "Qwen fast service is still loading the exact release"
                )
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenFastServiceAdapter:
    """Keep the fully gated full-GDN/DFlash release alive as a Fleet service."""

    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _expected_artifacts(sources: dict[str, str], prompt: bytes) -> dict[str, str]:
        model = MODEL_ARTIFACTS["fullgdn"]
        draft = DRAFT_ARTIFACTS["bf16"]
        return {
            "benchmark_manifest": EVIDENCE_SHA256["manifest"],
            "benchmark_preflight": EVIDENCE_SHA256["preflight"],
            "benchmark_quality": EVIDENCE_SHA256["quality"],
            "benchmark_result": EVIDENCE_SHA256["result"],
            "benchmark_speed": EVIDENCE_SHA256["speed"],
            "bare_engine_archive": ENGINE_ARCHIVE_SHA256,
            "dflash2_bf16_config": draft["config_sha256"],
            "dflash2_bf16_model": draft["model_sha256"],
            "dflash2_bf16_revision": draft["revision_sha256"],
            "image": IMAGE_ID.removeprefix("sha256:"),
            "model_fullgdn_manifest": model["manifest_sha256"],
            "model_fullgdn_sha256s": model["sha256s_sha256"],
            "prompt_bundle": hashlib.sha256(prompt).hexdigest(),
            "source_manifest": _canonical_sha256(sources),
        }

    @staticmethod
    def _verify_evidence() -> None:
        root = EVIDENCE_ROOT.lstat()
        if (
            not stat.S_ISDIR(root.st_mode)
            or root.st_uid != os.geteuid()
            or root.st_mode & 0o022
        ):
            raise QwenSpeedLabError("fast-release evidence root is unsafe")
        for key, name in EVIDENCE_FILES.items():
            path = EVIDENCE_ROOT / name
            item = path.lstat()
            if (
                not stat.S_ISREG(item.st_mode)
                or item.st_uid != os.geteuid()
                or item.st_nlink != 1
                or item.st_mode & 0o022
                or _sha256(path) != EVIDENCE_SHA256[key]
            ):
                raise QwenSpeedLabError("fast-release benchmark evidence changed")
        result = json.loads((EVIDENCE_ROOT / "result.json").read_text(encoding="utf-8"))
        speed = json.loads((EVIDENCE_ROOT / "speed.json").read_text(encoding="utf-8"))
        quality = json.loads((EVIDENCE_ROOT / "quality.json").read_text(encoding="utf-8"))
        if (
            result.get("terminal_success") is not True
            or result.get("variant") != WINNER_VARIANT
            or quality.get("passed") is not True
            or quality.get("successful_requests") != quality.get("request_count")
            or float(speed.get("median_decode_tps") or 0) < 130
            or float(speed.get("p95_warm_prefix_ttft_seconds") or 99) > 1
        ):
            raise QwenSpeedLabError("fast-release benchmark gates are not satisfied")

    @staticmethod
    def _profile_identity(
        context: RuntimeContext, sources: dict[str, str], prompt: bytes
    ) -> None:
        if (
            context.profile.profile_id != PROFILE_ID
            or context.profile.artifact_identity
            != AeonQwenFastServiceAdapter._expected_artifacts(sources, prompt)
        ):
            raise QwenSpeedLabError("fast-service profile artifact identity changed")
        lease = context.lease
        if (
            lease.host != HOST
            or lease.memory_total_mib is None
            or lease.memory_total_mib < 47 * 1024
            or abs(lease.vram_budget_gb - 41.25) > 1e-9
            or lease.exclusive is not True
            or context.scratch_path != lease.run_dir
            or PurePosixPath(str(lease.run_dir)).parent != REMOTE_RUN_ROOT
        ):
            raise QwenSpeedLabError("fast-service lease differs from its reviewed profile")

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if _RUNTIME_ID_RE.fullmatch(context.runtime_id) is None or context.job_id is not None:
            raise QwenSpeedLabError("fast-service runtime identity is malformed")
        variant = VARIANT_CONFIGS[WINNER_VARIANT]
        model = MODEL_ARTIFACTS[str(variant["model_id"])]
        draft = DRAFT_ARTIFACTS[str(variant["draft_id"])]
        sources = _source_manifest()
        prompt, _prompt_sources = _prompt_bundle()
        self._profile_identity(context, sources, prompt)
        self._verify_evidence()
        host_config = HOST_CONFIGS[HOST]
        scratch = str(context.scratch_path)
        source_root = f"{scratch}/source"
        request_path = f"{scratch}/speed-lab-request.json"
        before_device, _free, _inodes, before_allocated = _remote_metrics(
            HOST, scratch, create=True
        )
        AeonQwenSpeedLabAdapter._stage_sources(
            HOST, str(host_config["hostname"]), scratch, sources
        )
        local_prefix = context.run_dir / "system-prefix.txt"
        AeonQwenSpeedLabAdapter._write_private(local_prefix, prompt)
        AeonQwenSpeedLabAdapter._stage_file(
            HOST, local_prefix, f"{scratch}/system-prefix.txt"
        )
        request = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "runtime_id": context.runtime_id,
            "job_id": f"service-{context.runtime_id}",
            "host": HOST,
            "hostname": host_config["hostname"],
            "claim_id": context.lease.claim_id,
            "owner": context.lease.owner,
            "physical_gpu": context.lease.physical_gpu,
            "gpu_uuid": context.lease.gpu_uuid,
            "vram_budget_gb": context.lease.vram_budget_gb,
            "exclusive": context.lease.exclusive,
            "feature_dataset_bytes": None,
            "feature_dataset_path": None,
            "feature_dataset_rows": None,
            "feature_dataset_sha256": None,
            "scratch_path": scratch,
            "source_root": source_root,
            "source_files": sources,
            "model_id": variant["model_id"],
            "model_dir": model["model_dir"],
            "model_manifest_sha256": model["manifest_sha256"],
            "model_sha256s_sha256": model["sha256s_sha256"],
            "image_id": IMAGE_ID,
            "image_size_bytes": IMAGE_SIZE_BYTES,
            "engine_archive_sha256": ENGINE_ARCHIVE_SHA256,
            "engine_closure": None,
            "draft_id": variant["draft_id"],
            "draft_model_dir": draft["model_dir"],
            "draft_revision": draft["revision"],
            "draft_config_sha256": draft["config_sha256"],
            "draft_model_sha256": draft["model_sha256"],
            "container_name": f"aeon-speed-{context.runtime_id}",
            "port": PORT,
            "variant": WINNER_VARIANT,
            "runtime": {
                "attention_backend": variant["attention_backend"],
                "async_scheduling": False,
                "compilation_profile": variant["compilation_profile"],
                "context_tokens": variant["context_tokens"],
                "cuda_launch_blocking": False,
                "dspark_draft_topk": variant["dspark_draft_topk"],
                "enable_adaptive_verification": variant[
                    "enable_adaptive_verification"
                ],
                "enable_flashinfer_autotune": host_config[
                    "enable_flashinfer_autotune"
                ],
                "enable_prefix_caching": variant["enable_prefix_caching"],
                "enable_per_request_metrics": False,
                "feature_capture": False,
                "gdn_decode_kernel": "cuda",
                "gpu_memory_utilization": host_config["gpu_memory_utilization"],
                "kv_cache_dtype": variant["kv_cache_dtype"],
                "local_argmax_reduction": False,
                "mamba_cache_dtype": variant["mamba_cache_dtype"],
                "mamba_cache_mode": "align",
                "mamba_ssm_cache_dtype": variant["mamba_ssm_cache_dtype"],
                "max_batched_tokens": host_config["max_batched_tokens"],
                "max_num_seqs": 1,
                "model_runner": "v2",
                "nvfp4_a16": model["nvfp4_a16"],
                "relaxed_greedy_logit_margin": "0",
                "speculative_method": "dflash",
                "speculative_tokens": 7,
                "use_flashinfer_sampler": host_config["use_flashinfer_sampler"],
            },
            "benchmark": {
                "max_tokens": 512,
                "quality_repeats": 2,
                "repeats": 5,
                "sampling_profile": "aeon-greedy-medium",
            },
        }
        request_bytes = (
            json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        local_request = context.run_dir / "speed-lab-request.json"
        AeonQwenSpeedLabAdapter._write_private(local_request, request_bytes)
        AeonQwenSpeedLabAdapter._stage_file(HOST, local_request, request_path)
        context.heartbeat(None, "Qwen fast service exact artifact preflight")
        preflight = _remote_action(
            HOST,
            source_root,
            "preflight",
            request_path,
            request_sha256,
            timeout=1900,
        )
        if (
            preflight.get("image_id") != IMAGE_ID
            or preflight.get("engine_archive_sha256") != ENGINE_ARCHIVE_SHA256
            or preflight.get("engine_closure") is not None
            or preflight.get("enable_per_request_metrics") is not False
            or preflight.get("draft_model_sha256") != draft["model_sha256"]
            or preflight.get("model_manifest_sha256") != model["manifest_sha256"]
            or preflight.get("model_sha256s_sha256") != model["sha256s_sha256"]
            or preflight.get("prompt_fixture_sha256")
            != hashlib.sha256(prompt).hexdigest()
        ):
            raise QwenSpeedLabError("worker fast-service preflight identity changed")
        filesystem_id, free_bytes, free_inodes, allocated = _remote_metrics(
            HOST, scratch, create=False
        )
        if filesystem_id != before_device:
            raise QwenSpeedLabError("worker filesystem changed during fast-service staging")
        with self._lock:
            self._prepared[context.runtime_id] = {
                "host": HOST,
                "request_path": request_path,
                "request_sha256": request_sha256,
                "source_root": source_root,
            }
        return StoragePreparationResult(
            scratch_path=context.scratch_path,
            filesystem_id=filesystem_id,
            free_bytes_after_stage=free_bytes,
            free_inodes_after_stage=free_inodes,
            staged_bytes=max(0, allocated - before_allocated),
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError(
                "fast-service preflight receipt is absent", process_absent=True
            )
        result = _remote_action(
            HOST,
            prepared["source_root"],
            "service-spawn",
            prepared["request_path"],
            prepared["request_sha256"],
            timeout=60,
        )
        pid = result.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise QwenSpeedLabError("fast-service supervisor PID is malformed")
        try:
            with _ServiceHeartbeat(context, pid) as heartbeat:
                deadline = time.monotonic() + 2100
                while time.monotonic() < deadline:
                    heartbeat.check()
                    status = _remote_action(
                        HOST,
                        prepared["source_root"],
                        "service-status",
                        prepared["request_path"],
                        prepared["request_sha256"],
                        timeout=60,
                    )
                    if status.get("pid") not in {None, pid}:
                        raise QwenSpeedLabError("fast-service supervisor PID changed")
                    if status.get("state") == "ready":
                        _start_tunnel(
                            context.run_dir,
                            context.runtime_id,
                            prepared["request_sha256"],
                        )
                        heartbeat.check()
                        identity = (
                            f"aeon-qwen38-fast:{context.runtime_id}:"
                            f"{prepared['request_sha256']}:{pid}"
                        )
                        return LaunchResult(
                            pid=pid,
                            process_identity=identity,
                            endpoint=LOCAL_ENDPOINT,
                        )
                    if status.get("state") in {"absent", "unknown"}:
                        raise QwenSpeedLabError(
                            f"fast-service startup became {status.get('state')}"
                        )
                    time.sleep(5)
                raise QwenSpeedLabError("fast-service startup exceeded its bound")
        except BaseException as exc:
            try:
                status = _remote_action(
                    HOST,
                    prepared["source_root"],
                    "service-status",
                    prepared["request_path"],
                    prepared["request_sha256"],
                    timeout=60,
                )
            except Exception:
                raise
            if status.get("state") == "absent":
                raise AdapterLaunchError(
                    f"fast-service launch failed before a live process remained: {exc}",
                    process_absent=True,
                ) from exc
            raise

    @staticmethod
    def _runtime_identity(runtime: Mapping[str, Any]) -> tuple[str, str, int]:
        match = _PROCESS_IDENTITY_RE.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None
            or match.group(1) != runtime.get("runtime_id")
            or int(match.group(3)) != runtime.get("pid")
            or runtime.get("profile_id") != PROFILE_ID
            or runtime.get("host") != HOST
            or PurePosixPath(str(runtime.get("run_dir") or "")).parent
            != REMOTE_RUN_ROOT
        ):
            raise QwenSpeedLabError("fast-service runtime identity changed")
        return match.group(1), match.group(2), int(match.group(3))

    @classmethod
    def _runtime_action(
        cls, runtime: Mapping[str, Any], action: str, *, timeout: float = 120
    ) -> dict[str, Any]:
        runtime_id, digest, _pid = cls._runtime_identity(runtime)
        scratch = str(runtime["run_dir"])
        return _remote_action(
            HOST,
            f"{scratch}/source",
            action,
            f"{scratch}/speed-lab-request.json",
            digest,
            timeout=timeout,
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            runtime_id, digest, pid = self._runtime_identity(runtime)
            status = self._runtime_action(runtime, "service-status", timeout=60)
        except QwenSpeedLabTransportError:
            raise
        except QwenSpeedLabError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        if status.get("pid") not in {None, pid}:
            return ProbeResult(
                ProbeState.UNKNOWN, False, False, "fast-service PID identity changed"
            )
        if status.get("state") == "starting":
            return ProbeResult(
                ProbeState.STARTING, True, False, "Qwen fast service is starting"
            )
        if status.get("state") == "ready":
            path = Path(str(runtime["run_dir"])) / _TUNNEL_RECEIPT
            try:
                receipt = _private_json(path)
            except (FileNotFoundError, QwenSpeedLabError) as exc:
                return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
            if (
                receipt.get("runtime_id") != runtime_id
                or receipt.get("request_sha256") != digest
                or receipt.get("state") != "active"
                or not _tunnel_exact(receipt)
                or not _endpoint_ready()
            ):
                return ProbeResult(
                    ProbeState.UNKNOWN,
                    False,
                    False,
                    "fast-service loopback endpoint identity changed",
                )
            return ProbeResult(
                ProbeState.READY, True, False, "Qwen full-GDN fast release is ready"
            )
        if status.get("state") == "absent":
            return ProbeResult(
                ProbeState.ABSENT,
                False,
                True,
                str(status.get("failure") or "fast-service supervisor is absent")[:500],
            )
        return ProbeResult(
            ProbeState.UNKNOWN, False, False, "fast-service lifecycle is ambiguous"
        )

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            runtime_id, digest, _pid = self._runtime_identity(runtime)
            result = self._runtime_action(runtime, "service-stop", timeout=180)
            remote_absent = result.get("process_absent") is True
            tunnel_absent = _stop_tunnel(Path(str(runtime["run_dir"])), runtime_id, digest)
        except (QwenSpeedLabError, QwenSpeedLabTransportError) as exc:
            return StopResult(False, False, str(exc))
        absent = remote_absent and tunnel_absent
        return StopResult(
            absent,
            True,
            reason if absent else "fast-service exact processes are still stopping",
        )

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        scratch = storage.get("scratch_path")
        if scratch != runtime.get("run_dir") or not isinstance(scratch, str):
            raise QwenSpeedLabError("fast-service storage manifest identity changed")
        if runtime.get("process_identity") is None:
            local = Path(str(runtime["run_dir"])) / "speed-lab-request.json"
            digest = _sha256(local) if local.is_file() else None
            reclaimed = AeonQwenSpeedLabAdapter._cleanup_prelaunch_scratch(
                HOST, scratch, digest, str(runtime["runtime_id"])
            )
            return StorageFinalizationResult(
                True, True, reclaimed, "fast-service prelaunch scratch cleaned"
            )
        try:
            _remote_metrics(HOST, scratch, create=False)
        except FileNotFoundError:
            return StorageFinalizationResult(
                True, True, 0, "fast-service worker scratch already absent"
            )
        result = self._runtime_action(runtime, "service-cleanup", timeout=300)
        reclaimed = result.get("reclaimed_bytes")
        if (
            result.get("state") != "cleaned"
            or isinstance(reclaimed, bool)
            or not isinstance(reclaimed, int)
            or reclaimed < 0
        ):
            raise QwenSpeedLabError("fast-service cleanup receipt is malformed")
        return StorageFinalizationResult(
            True, True, reclaimed, "fast-service exact worker scratch removed"
        )


def create_fleet_adapter() -> AeonQwenFastServiceAdapter:
    return AeonQwenFastServiceAdapter()
