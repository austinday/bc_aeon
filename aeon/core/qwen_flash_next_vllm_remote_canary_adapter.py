"""Fleet-owned .179 GPU1 lane for the exact Flash-Next vLLM canary.

The worker is scratch-only: the canonical checkpoint and OCI archive remain on
``.177``; one immutable copy is staged into the unique Fleet attempt on ``.179``.
Qualification evidence is copied back and hash-settled before the exact attempt
directory can become cleanup eligible.  Nothing in this adapter selects a GPU,
touches a renter, or manages Docker state outside its task-owned container.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import subprocess
import threading
from typing import Any, Mapping

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    LaunchResult,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.core import qwen_flash_next_vllm_canary_adapter as local
from aeon.core.fleet_hosts import network_address
from aeon.scripts import qwen_flash_next_vllm_remote_worker as remote_worker


PROFILE_ID = remote_worker.PROFILE_ID
ADAPTER_NAME = "aeon-qwen38-flash-next-vllm-canary-179-v1"
HOST = remote_worker.HOST
HOSTNAME = remote_worker.HOSTNAME
PHYSICAL_GPU = remote_worker.PHYSICAL_GPU
RUN_ROOT = remote_worker.RUN_ROOT
CANONICAL_OUTPUT_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/artifacts/"
    "aeon-qwen38-flash-next-vllm-canary-179-gpu1"
)
REQUEST_NAME = "canary-request.json"
OWNERSHIP_NAME = "fleet-attempt-ownership.json"
REMOTE_PYTHON = "/usr/bin/python3"
REMOTE_DOCKER = "/home/aday/bin/docker"
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
TRANSFER_BYTES_PER_SECOND = 100_000_000
MODEL_BYTES_MAX = 137_000_000_000
IMAGE_BYTES_MAX = 8_700_000_000
SOURCE_BYTES_MAX = 20_000_000
STAGE_BYTES_MAX = MODEL_BYTES_MAX + IMAGE_BYTES_MAX + SOURCE_BYTES_MAX
RUNTIME_GROWTH_BYTES_MAX = 64_000_000_000
WORKER_FREE_RESERVE_BYTES = 20_000_000_000
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_PROCESS = re.compile(
    r"^aeon-vllm-canary-179:(fr-[0-9a-f]{32}):([0-9a-f]{64}):([0-9]+)$"
)

SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *local.SOURCE_FILES,
            "aeon/core/fleet_hosts.py",
            "aeon/scripts/qwen_flash_next_vllm_remote_worker.py",
        )
    )
)


class RemoteVllmCanaryError(RuntimeError):
    pass


class RemoteVllmCanaryTransportError(RemoteVllmCanaryError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source_receipts() -> dict[str, Mapping[str, Any]]:
    return {name: local._receipt(local.PACKAGE_ROOT / name) for name in SOURCE_FILES}


def expected_artifact_identity(payload: Mapping[str, str]) -> dict[str, str]:
    return {
        "adapter_source": _sha256(Path(__file__)),
        "remote_worker_source": _sha256(Path(remote_worker.__file__)),
        "shared_worker_source": _sha256(Path(local.worker.__file__)),
        "harness_source": _sha256(Path(local.harness.__file__)),
        "cuda_sampler_source": _sha256(Path(local.cuda_supervisor.__file__)),
        "runtime_contract_source": _sha256(Path(contract.__file__)),
        # Bind the exact shared canary closure independently from this lane's
        # additional wrapper, whose own digest is recorded above.
        "source_manifest": _canonical_sha(local._source_receipts()),
        "checkpoint_manifest": payload["checkpoint_manifest_sha256"],
        "derived_image": payload["derived_image_digest"].removeprefix("sha256:"),
        "derived_image_config": payload["derived_image_config_digest"],
        "derived_image_archive": payload["derived_image_archive_sha256"],
    }


def _ssh() -> list[str]:
    return [
        "/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes",
        "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o",
        "ControlPersist=no", "-o", "ServerAliveInterval=5", "-o",
        "ServerAliveCountMax=6", f"aday@{network_address(HOST)}",
    ]


def _remote_python(script: str, *args: str, timeout: float = 120) -> Mapping[str, Any]:
    command = [
        *_ssh(),
        shlex.join([
            "/usr/bin/env", "-i", "HOME=/home/aday",
            "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
            "LANG=C", "LC_ALL=C", LOW_PRIORITY, REMOTE_PYTHON, "-I", "-S", "-B",
            "-c", script, HOSTNAME, *args,
        ]),
    ]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RemoteVllmCanaryTransportError(".179 worker transport failed") from exc
    if result.returncode != 0 or len(result.stdout) > 1024 * 1024:
        raise RemoteVllmCanaryTransportError(".179 worker proof failed closed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteVllmCanaryTransportError(".179 worker proof was malformed") from exc
    if not isinstance(value, Mapping):
        raise RemoteVllmCanaryTransportError(".179 worker proof was not an object")
    return value


def _prepare_remote(runtime_id: str, token_sha256: str) -> Mapping[str, Any]:
    script = r'''
import json,os,pathlib,stat,sys
expected,runtime_id,token_sha=sys.argv[1:4]
assert os.uname().nodename==expected
assert len(runtime_id)==35 and runtime_id.startswith("fr-")
assert len(token_sha)==64
root=pathlib.Path("/home/aday/.local/state/fleet-compute/runs")
run=root/runtime_id
root.mkdir(mode=0o700,parents=True,exist_ok=True)
run.mkdir(mode=0o700)
for relative in ("source/aeon/core","source/aeon/scripts","source/aeon/behavioral_sft/data","assets","model","runtime-images"):
 (run/relative).mkdir(mode=0o700,parents=True)
for item in (root,run,*run.rglob("*")):
 m=item.lstat(); assert m.st_uid==os.geteuid() and not stat.S_ISLNK(m.st_mode)
 if stat.S_ISDIR(m.st_mode): item.chmod(0o700)
v=os.statvfs(run)
print(json.dumps({"filesystem_id":str(run.lstat().st_dev),"free_bytes":v.f_bavail*v.f_frsize,"free_inodes":v.f_favail,"token_sha256":token_sha},sort_keys=True))
'''
    return _remote_python(script, runtime_id, token_sha256)


def _stage(source: Path, destination: str, *, directory_contents: bool = False, timeout: float = 3600) -> None:
    transport = " ".join(_ssh()[:-1])
    source_arg = f"{source}/" if directory_contents else str(source)
    command = [
        LOW_PRIORITY, "/usr/bin/rsync", "-aH", "--checksum",
        "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", "--protect-args",
        f"--bwlimit={TRANSFER_BYTES_PER_SECOND // 1024}",
        "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
        "-e", transport, "--", source_arg,
        f"aday@{network_address(HOST)}:{destination}",
    ]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RemoteVllmCanaryTransportError("remote staging transport failed") from exc
    if result.returncode != 0:
        raise RemoteVllmCanaryTransportError("remote staging failed closed")


def _remote_metrics(runtime_id: str) -> Mapping[str, Any]:
    script = r'''
import json,os,pathlib,stat,sys
expected,runtime_id=sys.argv[1:3]; assert os.uname().nodename==expected
run=pathlib.Path("/home/aday/.local/state/fleet-compute/runs")/runtime_id
meta=run.lstat(); assert stat.S_ISDIR(meta.st_mode) and meta.st_uid==os.geteuid() and not meta.st_mode&0o077
total=0
for item in run.rglob("*"):
 m=item.lstat(); assert m.st_dev==meta.st_dev and m.st_uid==os.geteuid() and not stat.S_ISLNK(m.st_mode)
 if stat.S_ISREG(m.st_mode): total+=m.st_blocks*512
 elif not stat.S_ISDIR(m.st_mode): raise AssertionError
v=os.statvfs(run)
print(json.dumps({"filesystem_id":str(meta.st_dev),"free_bytes":v.f_bavail*v.f_frsize,"free_inodes":v.f_favail,"allocated_bytes":total},sort_keys=True))
'''
    return _remote_python(script, runtime_id)


def _remote_evidence_inventory(runtime_id: str) -> tuple[str, ...]:
    script = r'''
import json,os,pathlib,stat,sys
expected,runtime_id=sys.argv[1:3]; assert os.uname().nodename==expected
run=pathlib.Path("/home/aday/.local/state/fleet-compute/runs")/runtime_id
allowed={"source","assets","model","runtime-images","canary-request.json","fleet-attempt-ownership.json","output","mtp_off","mtp_on","status.json","supervisor.pid","supervisor.stdout","supervisor.stderr"}
names={item.name for item in run.iterdir()}; assert names<=allowed
wanted=[]
for name in ("output","mtp_off","mtp_on","status.json","supervisor.pid","supervisor.stdout","supervisor.stderr"):
 item=run/name
 if item.exists():
  meta=item.lstat(); assert meta.st_uid==os.geteuid() and not stat.S_ISLNK(meta.st_mode)
  assert stat.S_ISREG(meta.st_mode) or stat.S_ISDIR(meta.st_mode)
  wanted.append(name)
print(json.dumps({"items":wanted},sort_keys=True))
'''
    value = _remote_python(script, runtime_id)
    items = value.get("items")
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise RemoteVllmCanaryError("remote evidence inventory is malformed")
    return tuple(items)


def _settle_evidence(runtime_id: str, destination: Path) -> tuple[str, int]:
    items = _remote_evidence_inventory(runtime_id)
    if "status.json" not in items:
        raise RemoteVllmCanaryError("remote status evidence is absent")
    temporary = destination / ".evidence.settling"
    temporary.mkdir(mode=0o700)
    transport = " ".join(_ssh()[:-1])
    sources = [
        f"aday@{network_address(HOST)}:{RUN_ROOT / runtime_id / item}"
        for item in items
    ]
    command = [
        LOW_PRIORITY, "/usr/bin/rsync", "-aH", "--checksum", "--protect-args",
        "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
        f"--bwlimit={TRANSFER_BYTES_PER_SECOND // 1024}",
        "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
        "-e", transport, "--", *sources, str(temporary),
    ]
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        raise RemoteVllmCanaryTransportError("remote evidence settlement failed")
    lines: list[str] = []
    allocated = 0
    for item in sorted(temporary.rglob("*")):
        metadata = item.lstat()
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise RemoteVllmCanaryError("settled evidence contains an unsafe inode")
        if item.is_file():
            relative = item.relative_to(temporary).as_posix()
            lines.append(f"{_sha256(item)}  {relative}")
            allocated += metadata.st_blocks * 512
    raw = ("\n".join(lines) + "\n").encode("ascii")
    local._write_private(temporary / "SETTLED.sha256", raw)
    digest = hashlib.sha256(raw).hexdigest()
    os_replace_target = destination / "evidence"
    if os_replace_target.exists() or os_replace_target.is_symlink():
        raise RemoteVllmCanaryError("settled evidence target already exists")
    temporary.rename(os_replace_target)
    return digest, allocated


def _cleanup_remote_attempt(runtime_id: str, token: str) -> int:
    script = r'''
import hashlib,json,os,pathlib,shutil,stat,subprocess,sys
expected,runtime_id,token=sys.argv[1:4]; assert os.uname().nodename==expected
root=pathlib.Path("/home/aday/.local/state/fleet-compute/runs"); run=root/runtime_id
assert run.parent==root and runtime_id.startswith("fr-") and len(runtime_id)==35
meta=run.lstat(); assert stat.S_ISDIR(meta.st_mode) and meta.st_uid==os.geteuid() and not meta.st_mode&0o077
ownership=run/"fleet-attempt-ownership.json"; om=ownership.lstat()
assert stat.S_ISREG(om.st_mode) and om.st_uid==os.geteuid() and om.st_nlink==1 and not om.st_mode&0o077
receipt=json.loads(ownership.read_text()); assert receipt=={"schema_version":"aeon-fleet-worker-attempt-v1","runtime_id":runtime_id,"host":"192.168.0.179","physical_gpu":1,"worker_path":str(run),"token":token}
status=json.loads((run/"status.json").read_text()); assert status.get("state") in {"completed","failed"} and status.get("pid") is None
for arm in ("mtp_off","mtp_on"):
 name=f"aeon-vllm-{runtime_id}-{arm}"
 check=subprocess.run(["/home/aday/bin/fleet-low-priority","/home/aday/bin/docker","container","inspect",name],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=30,env={"HOME":"/home/aday","PATH":"/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin","LANG":"C","LC_ALL":"C"})
 assert check.returncode==1 and "No such" in check.stderr
allocated=0
for item in run.rglob("*"):
 m=item.lstat(); assert m.st_dev==meta.st_dev and m.st_uid==os.geteuid() and not stat.S_ISLNK(m.st_mode) and not item.is_mount()
 assert stat.S_ISREG(m.st_mode) or stat.S_ISDIR(m.st_mode)
 if stat.S_ISREG(m.st_mode): allocated+=m.st_blocks*512
shutil.rmtree(run)
assert not run.exists()
print(json.dumps({"removed_bytes":allocated,"token_sha256":hashlib.sha256(token.encode()).hexdigest()},sort_keys=True))
'''
    result = _remote_python(script, runtime_id, token, timeout=600)
    if result.get("token_sha256") != hashlib.sha256(token.encode()).hexdigest():
        raise RemoteVllmCanaryError("remote cleanup ownership token changed")
    removed = result.get("removed_bytes")
    if type(removed) is not int or removed < 0:
        raise RemoteVllmCanaryError("remote cleanup receipt is malformed")
    return removed


def _remote_action(runtime_id: str, action: str, digest: str, *, timeout: float = 120) -> Mapping[str, Any]:
    if action not in {"preflight", "spawn", "status", "stop"}:
        raise RemoteVllmCanaryError("remote action changed")
    run = RUN_ROOT / runtime_id
    command = [
        *_ssh(),
        shlex.join([
            "/usr/bin/env", "-i", "HOME=/home/aday",
            "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
            "LANG=C", "LC_ALL=C", f"PYTHONPATH={run}/source",
            "PYTHONDONTWRITEBYTECODE=1", LOW_PRIORITY, REMOTE_PYTHON,
            str(run / "source/aeon/scripts/qwen_flash_next_vllm_remote_worker.py"),
            action, str(run / REQUEST_NAME), digest,
        ]),
    ]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RemoteVllmCanaryTransportError("remote canary action transport failed") from exc
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteVllmCanaryTransportError("remote canary returned no receipt") from exc
    if (
        result.returncode != 0 or not isinstance(value, Mapping)
        or value.get("ok") is not True or not isinstance(value.get("result"), Mapping)
    ):
        detail = value.get("detail") if isinstance(value, Mapping) else "unknown"
        raise RemoteVllmCanaryError(f"remote canary {action} failed: {detail}")
    return value["result"]


class _Heartbeat:
    def __init__(self, context: RuntimeContext, detail: str) -> None:
        self.context = context
        self.detail = detail
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_Heartbeat":
        self.context.heartbeat(None, self.detail)
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise RemoteVllmCanaryError("remote canary heartbeat failed") from self.error

    def _run(self) -> None:
        while not self.stop.wait(60):
            try:
                self.context.heartbeat(None, self.detail)
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenFlashNextVllmRemoteCanaryAdapter:
    def __init__(self) -> None:
        self._prepared: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _payload(value: Mapping[str, Any]) -> dict[str, str]:
        return local.AeonQwenFlashNextVllmCanaryAdapter._payload(value)

    @staticmethod
    def _validate_context(context: RuntimeContext, payload: Mapping[str, str]) -> None:
        profile, lease = context.profile, context.lease
        expected_disk = (
            STAGE_BYTES_MAX + RUNTIME_GROWTH_BYTES_MAX
            + WORKER_FREE_RESERVE_BYTES + 999_999_999
        ) // 1_000_000_000
        placements = [item for item in profile.placements if item.enabled]
        if (
            profile.profile_id != PROFILE_ID or profile.project != PROFILE_ID
            or profile.enabled is not True or profile.adapter != ADAPTER_NAME
            or profile.mode.value != "batch" or profile.vram_budget_gb != contract.VRAM_CAP_GIB
            or profile.exclusive is not True or profile.stage_bytes_max != STAGE_BYTES_MAX
            or profile.runtime_growth_bytes_max != RUNTIME_GROWTH_BYTES_MAX
            or profile.worker_free_reserve_bytes != WORKER_FREE_RESERVE_BYTES
            or profile.min_disk_free_gb != expected_disk
            or profile.artifact_identity != expected_artifact_identity(payload)
            or len(placements) != 1 or placements[0].host != HOST
            or placements[0].physical_gpu != PHYSICAL_GPU
            or lease.host != HOST or lease.physical_gpu != PHYSICAL_GPU
            or lease.exclusive is not True or lease.vram_budget_gb != contract.VRAM_CAP_GIB
            or lease.memory_total_mib is None or lease.memory_total_mib < 94 * 1024
            or lease.memory_total_mib / 1024 - lease.vram_budget_gb < 6
            or context.job_id is None or context.scratch_path != lease.run_dir
            or PurePosixPath(str(lease.run_dir)) != RUN_ROOT / context.runtime_id
            or context.canonical_output_path != CANONICAL_OUTPUT_ROOT / context.runtime_id
            or context.cached_artifacts
        ):
            raise RemoteVllmCanaryError("remote canary profile/lease is not exact .179 GPU1")

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        payload = self._payload(context.payload)
        self._validate_context(context, payload)
        token = secrets.token_hex(32)
        token_sha = hashlib.sha256(token.encode()).hexdigest()
        root = context.canonical_output_path
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700)
        ownership = {
            "schema_version": "aeon-fleet-worker-attempt-v1",
            "runtime_id": context.runtime_id,
            "host": HOST,
            "physical_gpu": PHYSICAL_GPU,
            "worker_path": str(context.scratch_path),
            "token": token,
        }
        local._write_private(root / OWNERSHIP_NAME, json.dumps(ownership, sort_keys=True).encode() + b"\n")
        with _Heartbeat(context, "Staging exact vLLM model/image to .179 GPU1 scratch"):
            before = _prepare_remote(context.runtime_id, token_sha)
            run = RUN_ROOT / context.runtime_id
            for name in SOURCE_FILES:
                _stage(local.PACKAGE_ROOT / name, str(run / "source" / name), timeout=600)
            for name in local.ASSET_FILES:
                _stage(local.ASSET_ROOT / name, str(run / "assets" / name), timeout=600)
            _stage(Path(payload["checkpoint_path"]), str(run / "model"), directory_contents=True)
            archive_name = Path(payload["derived_image_archive_path"]).name
            remote_archive = run / "runtime-images" / archive_name
            _stage(Path(payload["derived_image_archive_path"]), str(remote_archive))
            source_receipts = _source_receipts()
            asset_receipts = {
                name: local._receipt(local.ASSET_ROOT / name) for name in local.ASSET_FILES
            }
            request = {
                "schema_version": local.worker.SCHEMA, "runtime_id": context.runtime_id,
                "job_id": context.job_id, "host": HOST, "hostname": HOSTNAME,
                "physical_gpu": PHYSICAL_GPU, "gpu_uuid": context.lease.gpu_uuid,
                "claim_id": context.lease.claim_id, "owner": context.lease.owner,
                "exclusive": True, "vram_cap_gib": contract.VRAM_CAP_GIB,
                "canonical_output_path": str(run), "checkpoint_path": str(run / "model"),
                "checkpoint_manifest_path": str(run / "model/SHA256SUMS"),
                "checkpoint_manifest_sha256": payload["checkpoint_manifest_sha256"],
                "derived_image_digest": payload["derived_image_digest"],
                "derived_image_config_digest": payload["derived_image_config_digest"],
                "derived_image_archive_path": str(remote_archive),
                "derived_image_archive_sha256": payload["derived_image_archive_sha256"],
                "served_model": contract.SERVED_MODEL, "runtime": contract.expected_runtime(),
                "source_files": source_receipts, "asset_files": asset_receipts,
            }
            raw = json.dumps(request, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
            digest = hashlib.sha256(raw).hexdigest()
            request_path = root / REQUEST_NAME
            local._write_private(request_path, raw)
            _stage(request_path, str(run / REQUEST_NAME), timeout=600)
            _stage(root / OWNERSHIP_NAME, str(run / OWNERSHIP_NAME), timeout=600)
            context.startup_check()
            preflight = _remote_action(context.runtime_id, "preflight", digest, timeout=2400)
            if preflight != {
                "request_sha256": digest,
                "checkpoint_manifest_sha256": payload["checkpoint_manifest_sha256"],
                "derived_image_digest": payload["derived_image_digest"],
                "derived_image_archive_sha256": payload["derived_image_archive_sha256"],
                "vram_cap_gib": contract.VRAM_CAP_GIB,
            }:
                raise RemoteVllmCanaryError("remote semantic preflight identity changed")
            after = _remote_metrics(context.runtime_id)
        if after["filesystem_id"] != before["filesystem_id"]:
            raise RemoteVllmCanaryError("remote scratch filesystem changed during staging")
        if int(after["allocated_bytes"]) > STAGE_BYTES_MAX:
            raise RemoteVllmCanaryError("remote staged bytes exceeded the reviewed ceiling")
        with self._lock:
            self._prepared[context.runtime_id] = (digest, token_sha)
        return StoragePreparationResult(
            context.scratch_path, str(after["filesystem_id"]), int(after["free_bytes"]),
            int(after["free_inodes"]), int(after["allocated_bytes"]),
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError("remote canary preflight is absent", process_absent=True)
        digest, _token = prepared
        try:
            with _Heartbeat(context, "Running exact vLLM MTP-off/on canary on .179 GPU1"):
                result = _remote_action(context.runtime_id, "spawn", digest, timeout=2400)
            pid = result.get("pid")
            if type(pid) is not int or pid <= 1:
                raise RemoteVllmCanaryError("remote canary PID is malformed")
            context.heartbeat(pid, "Remote vLLM qualification arms are running")
            return LaunchResult(pid, f"aeon-vllm-canary-179:{context.runtime_id}:{digest}:{pid}")
        except BaseException as exc:
            status = _remote_action(context.runtime_id, "status", digest, timeout=30)
            if status.get("state") in {"absent", "completed", "failed"}:
                raise AdapterLaunchError(f"remote canary launch failed: {exc}", process_absent=True) from exc
            raise

    @staticmethod
    def _identity(runtime: Mapping[str, Any]) -> tuple[str, str, int]:
        runtime_id = str(runtime.get("runtime_id") or "")
        match = _PROCESS.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None or match.group(1) != runtime_id
            or int(match.group(3)) != runtime.get("pid")
            or runtime.get("profile_id") != PROFILE_ID or runtime.get("host") != HOST
            or runtime.get("physical_gpu") != PHYSICAL_GPU
            or PurePosixPath(str(runtime.get("run_dir") or "")) != RUN_ROOT / runtime_id
        ):
            raise RemoteVllmCanaryError("saved remote canary identity changed")
        return runtime_id, match.group(2), int(match.group(3))

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            runtime_id, digest, pid = self._identity(runtime)
            status = _remote_action(runtime_id, "status", digest, timeout=30)
        except RemoteVllmCanaryTransportError:
            raise
        except RemoteVllmCanaryError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        state = status.get("state")
        if state == "running" and status.get("pid") == pid:
            return ProbeResult(ProbeState.RUNNING, True, False, ".179 GPU1 canary is running")
        if state == "completed":
            return ProbeResult(ProbeState.COMPLETED, False, True, "remote speed/MTP/semantic gates passed")
        if state == "failed":
            return ProbeResult(ProbeState.FAILED, False, True, str(status.get("failure") or "remote canary failed")[:500])
        if state == "absent":
            return ProbeResult(ProbeState.ABSENT, False, True, "remote canary supervisor absent")
        return ProbeResult(ProbeState.UNKNOWN, False, False, "remote lifecycle is ambiguous")

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            runtime_id, digest, _pid = self._identity(runtime)
            result = _remote_action(runtime_id, "stop", digest, timeout=120)
        except RemoteVllmCanaryError as exc:
            return StopResult(False, False, str(exc))
        absent = result.get("process_absent") is True
        return StopResult(absent, True, reason if absent else "remote canary is still stopping")

    def finalize_storage(self, runtime: Mapping[str, Any], storage: Mapping[str, Any]) -> StorageFinalizationResult:
        runtime_id, digest, _pid = self._identity(runtime)
        if storage.get("scratch_path") != runtime.get("run_dir"):
            raise RemoteVllmCanaryError("remote canary scratch identity changed")
        if runtime.get("process_absent") != 1:
            raise RemoteVllmCanaryError("remote canary process absence is unproven")
        status = _remote_action(runtime_id, "status", digest, timeout=30)
        if status.get("state") not in {"completed", "failed"} or status.get("pid") is not None:
            raise RemoteVllmCanaryError("remote terminal status is ambiguous")
        root = CANONICAL_OUTPUT_ROOT / runtime_id
        ownership = json.loads((root / OWNERSHIP_NAME).read_text(encoding="utf-8"))
        if (
            not isinstance(ownership, Mapping)
            or ownership.get("runtime_id") != runtime_id
            or ownership.get("host") != HOST
            or ownership.get("physical_gpu") != PHYSICAL_GPU
            or ownership.get("worker_path") != str(RUN_ROOT / runtime_id)
            or not isinstance(ownership.get("token"), str)
        ):
            raise RemoteVllmCanaryError("durable worker ownership receipt changed")
        settled_sha, _settled_bytes = _settle_evidence(runtime_id, root)
        evidence = root / "evidence"
        settled_status = json.loads((evidence / "status.json").read_text(encoding="utf-8"))
        if settled_status != status:
            raise RemoteVllmCanaryError("settled terminal status changed")
        if status.get("state") == "completed":
            qualification_path = evidence / "output/qualification.json"
            qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
            # Validate with remote placement bindings while preserving the exact
            # local performance/semantic/MTP promotion gates.
            original_host, original_gpu = contract.HOST, contract.PHYSICAL_GPU
            try:
                contract.HOST, contract.PHYSICAL_GPU = HOST, PHYSICAL_GPU
                failures = contract.validate_qualification_receipt(qualification)
            finally:
                contract.HOST, contract.PHYSICAL_GPU = original_host, original_gpu
            if failures:
                raise RemoteVllmCanaryError(
                    "settled qualification is not promotion compatible: " + "; ".join(failures)
                )
            manifest = evidence / "output/MANIFEST.sha256"
            if _sha256(manifest) != status.get("manifest_sha256"):
                raise RemoteVllmCanaryError("settled qualification manifest changed")
        settlement = {
            "schema_version": "aeon-qwen38-flash-next-vllm-remote-settlement-v1",
            "runtime_id": runtime_id,
            "host": HOST,
            "physical_gpu": PHYSICAL_GPU,
            "terminal_state": status["state"],
            "settled_manifest_sha256": settled_sha,
            "promotion_compatible": status["state"] == "completed",
        }
        local._write_private(
            root / "SETTLEMENT.json",
            json.dumps(settlement, sort_keys=True).encode() + b"\n",
        )
        removed = _cleanup_remote_attempt(runtime_id, str(ownership["token"]))
        return StorageFinalizationResult(
            True, True, removed,
            "remote evidence settled on canonical .177 and exact .179 attempt scratch removed",
        )


def create_fleet_adapter() -> AeonQwenFlashNextVllmRemoteCanaryAdapter:
    return AeonQwenFlashNextVllmRemoteCanaryAdapter()
