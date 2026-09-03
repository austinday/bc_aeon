"""Fail-closed long-lived Flash-Next service lane for Fleet worker ``.179``.

Fleet owns placement, leases, durable cache references, and eviction. This
adapter binds one selected exclusive lease to the qualified private release, a
pinned preloaded OCI image, a task-cgrouped Docker container on the worker, and
an exact local loopback SSH tunnel. It never chooses a GPU, calls the
coordinator, pulls an image, or deletes shared cache/image state.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import socket
import stat
import subprocess
import tarfile
import threading
import time
from typing import Any, Callable, Mapping

import requests

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.artifact_cache import ArtifactCacheSafetyError
from fleet_compute.models import (
    ArtifactCacheBinding,
    ArtifactCacheContract,
    ArtifactDescriptor,
    ArtifactKind,
    LaunchResult,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from aeon.scripts import qwen_flash_next_remote_service_worker as worker

from . import qwen_artifact_cache as cache_backend
from . import qwen_flash_next_service_adapter as service
from .fleet_hosts import network_address


PROFILE_ID = "aeon-qwen38-flash-next-179"
ADAPTER_NAME = "aeon-qwen38-flash-next-remote-service-v1"
HOST = worker.HOST
HOSTNAME = worker.HOSTNAME
RUN_ROOT = worker.RUN_ROOT
CACHE_ROOT = worker.CACHE_ROOT
VRAM_BUDGET_GB = worker.VRAM_BUDGET_GB
MIN_PHYSICAL_VRAM_GB = worker.MIN_PHYSICAL_VRAM_GB
LOCAL_PORT_BASE = 18040
REMOTE_PORT_BASE = 18140
CONTAINER_PORT = worker.CONTAINER_PORT
REMOTE_PYTHON = cache_backend.REMOTE_PYTHON
REMOTE_WRAPPER = cache_backend.REMOTE_WRAPPER
LOW_PRIORITY = str(cache_backend.FLEET_LOW_PRIORITY)

RELEASE_ARTIFACT_ID = "aeon-qwen38-flash-next-release"
MODEL_ARTIFACT_ID = "aeon-qwen38-flash-next-materialized-model"
IMAGE_ARTIFACT_ID = "aeon-qwen38-flash-next-image"
# These are reviewed build/settlement ceilings, not claims about an artifact
# that has not been built yet.  Promotion replaces them with exact validated
# SHA256SUMS inventory sizes and inode counts.
TREE_SIZE_BYTES_MAX = 150 * 1024**3
RELEASE_SIZE_BYTES_MAX = TREE_SIZE_BYTES_MAX
RELEASE_INODE_COUNT_MAX = 10_000
RELEASE_TRANSFER_BYTES_MAX = TREE_SIZE_BYTES_MAX
RELEASE_COLD_PEAK_BYTES_MAX = TREE_SIZE_BYTES_MAX
MODEL_SIZE_BYTES_MAX = TREE_SIZE_BYTES_MAX
MODEL_INODE_COUNT_MAX = 10_000
MODEL_TRANSFER_BYTES_MAX = TREE_SIZE_BYTES_MAX
MODEL_COLD_PEAK_BYTES_MAX = TREE_SIZE_BYTES_MAX
IMAGE_RECEIPT_BYTES_MAX = 65_536
IMAGE_INODE_COUNT_MAX = 1
IMAGE_TRANSFER_BYTES_MAX = 64_000_000_000
IMAGE_COLD_PEAK_BYTES_MAX = 128_000_000_000
CACHE_QUOTA_BYTES = (
    RELEASE_COLD_PEAK_BYTES_MAX
    + MODEL_COLD_PEAK_BYTES_MAX
    + IMAGE_COLD_PEAK_BYTES_MAX
)
CACHE_QUOTA_INODES = (
    RELEASE_INODE_COUNT_MAX + MODEL_INODE_COUNT_MAX + IMAGE_INODE_COUNT_MAX
)
STAGE_BYTES_MAX = CACHE_QUOTA_BYTES
RUNTIME_GROWTH_BYTES_MAX = 10_000_000_000
WORKER_FREE_RESERVE_BYTES = 20_000_000_000
MIN_DISK_FREE_GB = (
    STAGE_BYTES_MAX
    + RUNTIME_GROWTH_BYTES_MAX
    + WORKER_FREE_RESERVE_BYTES
    + 999_999_999
) // 1_000_000_000
MATERIALIZED_MODEL_DIR = service.MATERIALIZED_MODEL_DIR
MATERIALIZATION_RECEIPT = service.MATERIALIZATION_RECEIPT
REMOTE_CONTRACT_SCHEMA = "aeon-qwen38-flash-next-remote-staging-v1"
REMOTE_GATE_SCHEMA = "aeon-qwen38-flash-next-remote-service-ready-v1"
PROCESS_PREFIX = "aeon-flash-next-remote"
TUNNEL_RECEIPT = "flash-next-remote-tunnel.json"
PRELAUNCH_CLEANUP_RECEIPT = "flash-next-remote-prelaunch-cleanup.json"
REQUEST_NAME = "flash-next-remote-request.json"
WORKER_SOURCE = Path(worker.__file__).resolve()
CANONICAL_IMAGE_ARCHIVE = Path(
    "/home/aday/.local/state/aeon-flash-next/runtime-images/"
    "qwen38-flash-next-sm120-headroom-a6c61-424e.oci.tar"
)
# These annotations and the archive size are independently read back from the
# exact headroom build; the archive, manifest, config, and local Docker ID remain
# separate identities throughout staging.
IMAGE_ARCHIVE_SIZE_BYTES = 13_951_062_528
IMAGE_OCI_NAME = (
    "docker.io/aeon/sglang:qwen38-flash-next-sm120-headroom-a6c61-424e"
)
IMAGE_OCI_REF_NAME = "qwen38-flash-next-sm120-headroom-a6c61-424e"
_OCI_METADATA_MAX_BYTES = 16 * 1024 * 1024
_OCI_MAX_MEMBERS = 100_000

RUNTIME_ENVIRONMENT = dict(service.CONSTANT_RUNTIME_ENV)

_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER = re.compile(r"^[0-9a-f]{64}$")
_PROCESS = re.compile(
    r"^aeon-flash-next-remote:(fr-[0-9a-f]{32}):([0-9a-f]{64}):"
    r"([0-9a-f]{64}):([0-9a-f]{64}):([0-9]+):([0-9]+):([0-9]+):([0-9]+)$"
)


class RemoteFlashNextServiceError(RuntimeError):
    """The remote artifact, lease, process, or tunnel identity failed closed."""


class RemoteFlashNextTransportError(RemoteFlashNextServiceError):
    """The selected worker could not be reached; absence is not implied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def adapter_source_sha256() -> str:
    return _sha256(Path(__file__).resolve())


def worker_source_sha256() -> str:
    return _sha256(WORKER_SOURCE)


def remote_staging_contract() -> dict[str, Any]:
    """Return the immutable, release-independent worker staging contract."""

    return {
        "schema_version": REMOTE_CONTRACT_SCHEMA,
        "profile_id": PROFILE_ID,
        "host": HOST,
        "hostname": HOSTNAME,
        "worker_source_sha256": worker_source_sha256(),
        "artifact_cache_backend_source_sha256": _sha256(
            Path(cache_backend.__file__).resolve()
        ),
        "production_binding_validator_source_sha256": _sha256(
            Path(service.__file__).resolve()
        ),
        "run_root": str(RUN_ROOT),
        "cache_root": str(CACHE_ROOT),
        "release": {
            "artifact_id": RELEASE_ARTIFACT_ID,
            "canonical_path": str(service.RELEASE_DIR),
            "manifest_path": str(service.RELEASE_DIR / "SHA256SUMS"),
            "size_bytes_max": RELEASE_SIZE_BYTES_MAX,
            "inode_count_max": RELEASE_INODE_COUNT_MAX,
            "transfer_bytes_max": RELEASE_TRANSFER_BYTES_MAX,
            "cold_peak_bytes_max": RELEASE_COLD_PEAK_BYTES_MAX,
            "rsync_archive_hardlinks": True,
        },
        "materialized_model": {
            "artifact_id": MODEL_ARTIFACT_ID,
            "canonical_path": str(MATERIALIZED_MODEL_DIR),
            "manifest_path": str(MATERIALIZED_MODEL_DIR / "SHA256SUMS"),
            "completion_receipt_path": str(MATERIALIZATION_RECEIPT),
            "size_bytes_max": MODEL_SIZE_BYTES_MAX,
            "inode_count_max": MODEL_INODE_COUNT_MAX,
            "transfer_bytes_max": MODEL_TRANSFER_BYTES_MAX,
            "cold_peak_bytes_max": MODEL_COLD_PEAK_BYTES_MAX,
            "mount_path": "/model",
            "mount_read_only": True,
            "source_role": "offline-materialized-canonical-checkpoint",
            "rsync_archive_hardlinks": True,
        },
        "image": {
            "artifact_id": IMAGE_ARTIFACT_ID,
            "oci_manifest_digest_sha256": (
                service.SGLANG_IMAGE_DIGEST.removeprefix("sha256:")
            ),
            "oci_config_digest_sha256": (
                service.SGLANG_IMAGE_CONFIG_DIGEST.removeprefix("sha256:")
            ),
            "local_docker_image_id_sha256": service.SGLANG_IMAGE_ID.removeprefix(
                "sha256:"
            ),
            "oci_archive_sha256": service.SGLANG_IMAGE_ARCHIVE_SHA256,
            "receipt_bytes_max": IMAGE_RECEIPT_BYTES_MAX,
            "transfer_bytes_max": IMAGE_TRANSFER_BYTES_MAX,
            "cold_peak_bytes_max": IMAGE_COLD_PEAK_BYTES_MAX,
            "pull": False,
            "global_image_cleanup": False,
        },
        "lease": {
            "exclusive": True,
            "vram_budget_gb": VRAM_BUDGET_GB,
            "physical_reserve_gib_min": 6,
            "hard_physical_gpu": False,
        },
        "runtime_environment": dict(sorted(RUNTIME_ENVIRONMENT.items())),
        "ports": {
            "local_base": LOCAL_PORT_BASE,
            "remote_base": REMOTE_PORT_BASE,
            "container": CONTAINER_PORT,
        },
        "cleanup": {
            "run_scratch_only": True,
            "cache_references_owned_by_fleet": True,
            "canonical_output_removed": False,
        },
    }


def remote_staging_contract_sha256() -> str:
    return _canonical_sha(remote_staging_contract())


def remote_artifact_identity(
    binding: service.ProductionBinding,
) -> dict[str, str]:
    identity = dict(binding.artifact_identity)
    identity["adapter_source"] = adapter_source_sha256()
    identity["remote_staging_contract"] = remote_staging_contract_sha256()
    return identity


def blocked_gate() -> dict[str, object]:
    """Compatibility readback: implementation is complete but profile is inert."""

    return {
        "schema_version": REMOTE_GATE_SCHEMA,
        "profile_id": PROFILE_ID,
        "blocked": False,
        "reason": "reviewed adapter is present; checked-in profile remains disabled until atomic qualification promotion",
        "live_actions_permitted": False,
        "remote_staging_contract": remote_staging_contract_sha256(),
    }


def _tree_inventory(
    root: Path,
    *,
    expected_digest: str,
    maximum_bytes: int,
    maximum_inodes: int,
) -> tuple[int, int]:
    metadata = root.lstat()
    manifest = root / "SHA256SUMS"
    manifest_metadata = manifest.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or not stat.S_ISREG(manifest_metadata.st_mode)
        or manifest_metadata.st_uid != os.geteuid()
        or manifest_metadata.st_nlink != 1
        or manifest_metadata.st_mode & 0o077
        or not 0 < manifest_metadata.st_size <= 16 * 1024 * 1024
        or _sha256(manifest) != expected_digest
    ):
        raise RemoteFlashNextServiceError("canonical tree/manifest identity changed")
    records: set[str] = set()
    previous = ""
    total = manifest_metadata.st_size
    for raw in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(
            r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)", raw
        )
        if match is None or match.group(2) <= previous:
            raise RemoteFlashNextServiceError("canonical SHA256SUMS is malformed")
        name = match.group(2)
        item = root / name
        item_metadata = item.lstat()
        if (
            not stat.S_ISREG(item_metadata.st_mode)
            or item_metadata.st_uid != os.geteuid()
            or item_metadata.st_mode & 0o077
        ):
            raise RemoteFlashNextServiceError("canonical tree member is unsafe")
        total += item_metadata.st_size
        records.add(name)
        previous = name
    actual = {item.name for item in root.iterdir() if item.name != "SHA256SUMS"}
    inodes = 1 + len(records) + 1
    if (
        actual != records
        or not records
        or total > maximum_bytes
        or inodes > maximum_inodes
    ):
        raise RemoteFlashNextServiceError("canonical tree inventory exceeds its closure")
    return total, inodes


def _artifact_cache_contract(
    *,
    release_tree_sha256: str,
    checkpoint_tree_sha256: str,
    release_size_bytes: int,
    release_inode_count: int,
    model_size_bytes: int,
    model_inode_count: int,
) -> dict[str, Any]:
    for digest in (release_tree_sha256, checkpoint_tree_sha256):
        if _SHA.fullmatch(digest) is None:
            raise RemoteFlashNextServiceError(
                "remote cache identity is not a SHA-256 digest"
            )
    if (
        type(release_size_bytes) is not int
        or not 1 <= release_size_bytes <= RELEASE_SIZE_BYTES_MAX
        or type(release_inode_count) is not int
        or not 1 <= release_inode_count <= RELEASE_INODE_COUNT_MAX
        or type(model_size_bytes) is not int
        or not 1 <= model_size_bytes <= MODEL_SIZE_BYTES_MAX
        or type(model_inode_count) is not int
        or not 1 <= model_inode_count <= MODEL_INODE_COUNT_MAX
    ):
        raise RemoteFlashNextServiceError("remote cache inventory is outside its bound")
    image = service.SGLANG_IMAGE_ID.removeprefix("sha256:")
    cold = release_size_bytes + model_size_bytes + IMAGE_COLD_PEAK_BYTES_MAX
    return {
        "worker_cache_root": str(CACHE_ROOT),
        "quota_bytes": cold,
        "quota_inodes": release_inode_count + model_inode_count + IMAGE_INODE_COUNT_MAX,
        "entry_idle_ttl_seconds": 86_400,
        "transfer_concurrency": 1,
        "transfer_bytes_per_second": 100_000_000,
        "artifacts": [
            {
                "artifact_id": RELEASE_ARTIFACT_ID,
                "identity_key": "release_tree",
                "kind": "manifested_tree",
                "canonical_path": str(service.RELEASE_DIR),
                "digest_sha256": release_tree_sha256,
                "size_bytes_max": release_size_bytes,
                "inode_count_max": release_inode_count,
                "transfer_bytes_max": release_size_bytes,
                "cold_peak_bytes_max": release_size_bytes,
                "manifest_path": str(service.RELEASE_DIR / "SHA256SUMS"),
                "manifest_format": "sha256sum-v1",
            },
            {
                "artifact_id": MODEL_ARTIFACT_ID,
                "identity_key": "materialized_checkpoint_tree",
                "kind": "manifested_tree",
                "canonical_path": str(MATERIALIZED_MODEL_DIR),
                "digest_sha256": checkpoint_tree_sha256,
                "size_bytes_max": model_size_bytes,
                "inode_count_max": model_inode_count,
                "transfer_bytes_max": model_size_bytes,
                "cold_peak_bytes_max": model_size_bytes,
                "manifest_path": str(MATERIALIZED_MODEL_DIR / "SHA256SUMS"),
                "manifest_format": "sha256sum-v1",
            },
            {
                "artifact_id": IMAGE_ARTIFACT_ID,
                "identity_key": "image_local_id",
                "kind": "oci_archive",
                "canonical_path": str(CANONICAL_IMAGE_ARCHIVE),
                "digest_sha256": image,
                "size_bytes_max": IMAGE_RECEIPT_BYTES_MAX,
                "inode_count_max": IMAGE_INODE_COUNT_MAX,
                "transfer_bytes_max": IMAGE_TRANSFER_BYTES_MAX,
                "cold_peak_bytes_max": IMAGE_COLD_PEAK_BYTES_MAX,
                "manifest_path": None,
                "manifest_format": None,
            },
        ],
    }


def validate_promoted_artifact_cache(
    raw: Mapping[str, Any], artifact_identity: Mapping[str, str]
) -> int:
    """Validate the exact three-object promoted cache and return its cold peak."""

    try:
        contract = ArtifactCacheContract.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteFlashNextServiceError(
            "promoted remote cache contract is malformed"
        ) from exc
    if contract.to_dict() != dict(raw):
        raise RemoteFlashNextServiceError(
            "promoted remote cache contract is not canonical"
        )
    artifacts = list(contract.artifacts)
    if [item.artifact_id for item in artifacts] != [
        RELEASE_ARTIFACT_ID,
        MODEL_ARTIFACT_ID,
        IMAGE_ARTIFACT_ID,
    ]:
        raise RemoteFlashNextServiceError("promoted remote cache order changed")
    release, model, image = artifacts
    image_digest = service.SGLANG_IMAGE_ID.removeprefix("sha256:")
    if (
        contract.worker_cache_root != str(CACHE_ROOT)
        or contract.entry_idle_ttl_seconds != 86_400
        or contract.transfer_concurrency != 1
        or contract.transfer_bytes_per_second != 100_000_000
        or release.identity_key != "release_tree"
        or release.kind is not ArtifactKind.MANIFESTED_TREE
        or Path(release.canonical_path) != service.RELEASE_DIR
        or release.digest_sha256 != artifact_identity.get("release_tree")
        or Path(str(release.manifest_path)) != service.RELEASE_DIR / "SHA256SUMS"
        or release.manifest_format != "sha256sum-v1"
        or not 1 <= release.size_bytes_max <= RELEASE_SIZE_BYTES_MAX
        or release.inode_count_max > RELEASE_INODE_COUNT_MAX
        or release.transfer_bytes_max != release.size_bytes_max
        or release.cold_peak_bytes_max != release.size_bytes_max
        or model.identity_key != "materialized_checkpoint_tree"
        or model.kind is not ArtifactKind.MANIFESTED_TREE
        or Path(model.canonical_path) != MATERIALIZED_MODEL_DIR
        or model.digest_sha256
        != artifact_identity.get("materialized_checkpoint_tree")
        or Path(str(model.manifest_path))
        != MATERIALIZED_MODEL_DIR / "SHA256SUMS"
        or model.manifest_format != "sha256sum-v1"
        or not 1 <= model.size_bytes_max <= MODEL_SIZE_BYTES_MAX
        or model.inode_count_max > MODEL_INODE_COUNT_MAX
        or model.transfer_bytes_max != model.size_bytes_max
        or model.cold_peak_bytes_max != model.size_bytes_max
        or image.identity_key != "image_local_id"
        or image.kind is not ArtifactKind.OCI_ARCHIVE
        or Path(image.canonical_path) != CANONICAL_IMAGE_ARCHIVE
        or image.digest_sha256 != image_digest
        or image.digest_sha256 != artifact_identity.get("image_local_id")
        or image.size_bytes_max != IMAGE_RECEIPT_BYTES_MAX
        or image.inode_count_max != IMAGE_INODE_COUNT_MAX
        or image.transfer_bytes_max != IMAGE_TRANSFER_BYTES_MAX
        or image.cold_peak_bytes_max != IMAGE_COLD_PEAK_BYTES_MAX
        or image.manifest_path is not None
        or image.manifest_format is not None
    ):
        raise RemoteFlashNextServiceError(
            "promoted remote cache artifact identity changed"
        )
    cold_peak = sum(item.cold_peak_bytes_max for item in artifacts)
    inode_peak = sum(item.inode_count_max for item in artifacts)
    if (
        contract.quota_bytes != cold_peak
        or contract.quota_inodes != inode_peak
        or cold_peak > STAGE_BYTES_MAX
        or inode_peak > CACHE_QUOTA_INODES
    ):
        raise RemoteFlashNextServiceError(
            "promoted remote cache peak accounting changed"
        )
    return cold_peak


def promoted_artifact_cache(
    binding: service.ProductionBinding,
) -> dict[str, Any]:
    materialized_dir = Path(str(getattr(binding, "materialized_model_dir", "")))
    materialized_tree = str(
        getattr(binding, "materialized_checkpoint_tree_sha256", "")
    )
    if materialized_dir != MATERIALIZED_MODEL_DIR or materialized_tree != (
        binding.checkpoint_tree_sha256
    ) or binding.materialization_receipt != MATERIALIZATION_RECEIPT:
        raise RemoteFlashNextServiceError("materialized production binding changed")
    receipt_sha256 = str(getattr(binding, "materialization_receipt_sha256", ""))
    if (
        _SHA.fullmatch(receipt_sha256) is None
        or _sha256(MATERIALIZATION_RECEIPT) != receipt_sha256
    ):
        raise RemoteFlashNextServiceError("materialization completion receipt changed")
    release_size, release_inodes = _tree_inventory(
        binding.release_dir,
        expected_digest=binding.release_tree_sha256,
        maximum_bytes=RELEASE_SIZE_BYTES_MAX,
        maximum_inodes=RELEASE_INODE_COUNT_MAX,
    )
    model_size, model_inodes = _tree_inventory(
        materialized_dir,
        expected_digest=materialized_tree,
        maximum_bytes=MODEL_SIZE_BYTES_MAX,
        maximum_inodes=MODEL_INODE_COUNT_MAX,
    )
    if (
        release_size != binding.release_size_bytes
        or release_inodes != binding.release_inode_count + 1
        or model_size != binding.materialized_model_size_bytes
        or model_inodes != binding.materialized_model_inode_count + 1
    ):
        raise RemoteFlashNextServiceError("production tree inventory changed")
    return _artifact_cache_contract(
        release_tree_sha256=binding.release_tree_sha256,
        checkpoint_tree_sha256=materialized_tree,
        release_size_bytes=release_size,
        release_inode_count=release_inodes,
        model_size_bytes=model_size,
        model_inode_count=model_inodes,
    )


def promoted_artifact_cache_for_release(
    release_tree_sha256: str,
    checkpoint_tree_sha256: str | None = None,
) -> dict[str, Any]:
    if checkpoint_tree_sha256 is None:
        if release_tree_sha256 != service.ZERO_SHA256:
            raise RemoteFlashNextServiceError(
                "promoted materialized checkpoint identity is required"
            )
        checkpoint_tree_sha256 = service.ZERO_SHA256
    return _artifact_cache_contract(
        release_tree_sha256=release_tree_sha256,
        checkpoint_tree_sha256=checkpoint_tree_sha256,
        release_size_bytes=RELEASE_SIZE_BYTES_MAX,
        release_inode_count=RELEASE_INODE_COUNT_MAX,
        model_size_bytes=MODEL_SIZE_BYTES_MAX,
        model_inode_count=MODEL_INODE_COUNT_MAX,
    )


def _validate_flash_oci_layout_fd(
    descriptor: int,
    *,
    manifest_digest: str,
    config_digest: str,
    archive_size: int,
    expected_labels: Mapping[str, str],
    allowed_link_counts: frozenset[int],
) -> None:
    """Validate the OCI layout without substituting one image identity for another."""

    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or metadata.st_nlink not in allowed_link_counts
        or metadata.st_size != archive_size
    ):
        raise ArtifactCacheSafetyError("canonical Flash-Next OCI archive is unsafe")
    manifest_name = f"blobs/sha256/{manifest_digest}"
    config_name = f"blobs/sha256/{config_digest}"
    wanted_payloads = {"oci-layout", "index.json", manifest_name, config_name}
    payloads: dict[str, bytes] = {}
    members: dict[str, tarfile.TarInfo] = {}
    try:
        stream_fd = os.dup(descriptor)
        os.lseek(stream_fd, 0, os.SEEK_SET)
        with os.fdopen(stream_fd, "rb", closefd=True) as stream, tarfile.open(
            fileobj=stream, mode="r:"
        ) as bundle:
            for member in bundle:
                name = member.name.rstrip("/")
                parts = PurePosixPath(name).parts
                if (
                    not name
                    or name.startswith("/")
                    or ".." in parts
                    or name in members
                    or not (member.isfile() or member.isdir())
                    or member.islnk()
                    or member.issym()
                    or member.size < 0
                    or member.size > archive_size
                ):
                    raise ArtifactCacheSafetyError(
                        "canonical Flash-Next OCI archive has an unsafe member"
                    )
                members[name] = member
                if len(members) > _OCI_MAX_MEMBERS:
                    raise ArtifactCacheSafetyError(
                        "canonical Flash-Next OCI archive member bound changed"
                    )
                if name in wanted_payloads:
                    if (
                        not member.isfile()
                        or not 0 < member.size <= _OCI_METADATA_MAX_BYTES
                    ):
                        raise ArtifactCacheSafetyError(
                            "canonical Flash-Next OCI metadata is unsafe"
                        )
                    extracted = bundle.extractfile(member)
                    if extracted is None:
                        raise ArtifactCacheSafetyError(
                            "canonical Flash-Next OCI metadata is absent"
                        )
                    payload = extracted.read(_OCI_METADATA_MAX_BYTES + 1)
                    if len(payload) != member.size:
                        raise ArtifactCacheSafetyError(
                            "canonical Flash-Next OCI metadata changed"
                        )
                    payloads[name] = payload
        if set(payloads) != wanted_payloads:
            raise ArtifactCacheSafetyError(
                "canonical Flash-Next OCI identity closure is absent"
            )
        layout = json.loads(payloads["oci-layout"].decode("utf-8"))
        index = json.loads(payloads["index.json"].decode("utf-8"))
        manifest_payload = payloads[manifest_name]
        config_payload = payloads[config_name]
        manifest = json.loads(manifest_payload.decode("utf-8"))
        config = json.loads(config_payload.decode("utf-8"))
    except ArtifactCacheSafetyError:
        raise
    except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCacheSafetyError(
            "canonical Flash-Next OCI archive is malformed"
        ) from exc

    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    config_sha = hashlib.sha256(config_payload).hexdigest()
    if layout != {"imageLayoutVersion": "1.0.0"}:
        raise ArtifactCacheSafetyError("canonical Flash-Next OCI layout changed")
    if (
        not isinstance(index, dict)
        or set(index) != {"schemaVersion", "mediaType", "manifests"}
        or index.get("schemaVersion") != 2
        or index.get("mediaType") != "application/vnd.oci.image.index.v1+json"
        or not isinstance(index.get("manifests"), list)
        or len(index["manifests"]) != 1
    ):
        raise ArtifactCacheSafetyError("canonical Flash-Next OCI index changed")
    image = index["manifests"][0]
    if (
        not isinstance(image, dict)
        or set(image)
        != {"mediaType", "digest", "size", "annotations", "platform"}
        or image.get("mediaType")
        != "application/vnd.oci.image.manifest.v1+json"
        or image.get("digest") != f"sha256:{manifest_digest}"
        or image.get("size") != len(manifest_payload)
        or not isinstance(image.get("annotations"), dict)
        or image["annotations"].get("io.containerd.image.name") != IMAGE_OCI_NAME
        or image["annotations"].get("org.opencontainers.image.ref.name")
        != IMAGE_OCI_REF_NAME
        or image.get("platform") != {"architecture": "amd64", "os": "linux"}
        or manifest_sha != manifest_digest
    ):
        raise ArtifactCacheSafetyError(
            "canonical Flash-Next OCI manifest descriptor changed"
        )
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schemaVersion", "mediaType", "config", "layers"}
        or manifest.get("schemaVersion") != 2
        or manifest.get("mediaType")
        != "application/vnd.oci.image.manifest.v1+json"
        or not isinstance(manifest.get("config"), dict)
        or manifest["config"]
        != {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": f"sha256:{config_digest}",
            "size": len(config_payload),
        }
        or config_sha != config_digest
        or not isinstance(manifest.get("layers"), list)
        or not manifest["layers"]
    ):
        raise ArtifactCacheSafetyError(
            "canonical Flash-Next OCI manifest/config identity changed"
        )
    layers = manifest["layers"]
    layer_names: set[str] = set()
    for layer in layers:
        if (
            not isinstance(layer, dict)
            or set(layer) != {"mediaType", "digest", "size"}
            or layer.get("mediaType")
            != "application/vnd.oci.image.layer.v1.tar+gzip"
            or not isinstance(layer.get("digest"), str)
            or not layer["digest"].startswith("sha256:")
            or _SHA.fullmatch(layer["digest"].removeprefix("sha256:")) is None
            or isinstance(layer.get("size"), bool)
            or not isinstance(layer.get("size"), int)
            or not 0 < layer["size"] <= archive_size
        ):
            raise ArtifactCacheSafetyError(
                "canonical Flash-Next OCI layer descriptor changed"
            )
        layer_name = f"blobs/sha256/{layer['digest'].removeprefix('sha256:')}"
        layer_member = members.get(layer_name)
        if (
            layer_member is None
            or not layer_member.isfile()
            or layer_member.size != layer["size"]
        ):
            raise ArtifactCacheSafetyError(
                "canonical Flash-Next OCI layer closure changed"
            )
        layer_names.add(layer_name)
    regular = {name for name, member in members.items() if member.isfile()}
    directories = {name for name, member in members.items() if member.isdir()}
    if regular != wanted_payloads | layer_names or directories != {
        "blobs",
        "blobs/sha256",
    }:
        raise ArtifactCacheSafetyError(
            "canonical Flash-Next OCI blob closure changed"
        )
    rootfs = config.get("rootfs") if isinstance(config, dict) else None
    labels = config.get("config", {}).get("Labels") if isinstance(config, dict) else None
    if (
        config.get("architecture") != "amd64"
        or config.get("os") != "linux"
        or not isinstance(rootfs, dict)
        or rootfs.get("type") != "layers"
        or not isinstance(rootfs.get("diff_ids"), list)
        or len(rootfs["diff_ids"]) != len(layers)
        or any(
            not isinstance(item, str)
            or not item.startswith("sha256:")
            or _SHA.fullmatch(item.removeprefix("sha256:")) is None
            for item in rootfs["diff_ids"]
        )
        or not isinstance(labels, dict)
        or any(labels.get(key) != value for key, value in expected_labels.items())
    ):
        raise ArtifactCacheSafetyError(
            "canonical Flash-Next OCI config semantics changed"
        )


class FlashNextArtifactBackend(cache_backend.AeonQwenArtifactBackend):
    """Fleet cache backend for the release, materialized model, and image."""

    WORKER_CACHE_ROOT = CACHE_ROOT

    @staticmethod
    def _host(host: str) -> str:
        if host != HOST:
            raise ArtifactCacheSafetyError("Flash-Next cache host is not .179")
        return HOSTNAME

    @staticmethod
    def _ssh(host: str) -> list[str]:
        FlashNextArtifactBackend._host(host)
        return [
            "/usr/bin/ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "ControlPersist=no",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=6",
            f"aday@{network_address(host)}",
        ]

    def _remote_image_inspection(
        self, host: str, image_id: str
    ) -> dict[str, Any] | None:
        if image_id != service.SGLANG_IMAGE_ID:
            raise ArtifactCacheSafetyError(
                "Flash-Next local image ID was replaced by another identity"
            )
        image = super()._remote_image_inspection(host, image_id)
        if image is None:
            return None
        descriptor = image.get("Descriptor")
        repo_digests = image.get("RepoDigests")
        config = image.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("digest") != service.SGLANG_IMAGE_DIGEST
            or not isinstance(repo_digests, list)
            or worker.SGLANG_IMAGE_REPO_DIGEST not in repo_digests
            or not isinstance(labels, Mapping)
            or any(
                labels.get(key) != value
                for key, value in service.runtime_contract.EXPECTED_IMAGE_LABELS.items()
            )
        ):
            raise ArtifactCacheSafetyError(
                "Flash-Next remote manifest, repository, or label identity changed"
            )
        return image

    def _canonical_flash_archive(
        self,
        descriptor: ArtifactDescriptor,
        *,
        progress: Callable[[int, int], None],
    ) -> tuple[Path, str]:
        self._validate_descriptor(descriptor)
        archive = Path(descriptor.canonical_path)
        if archive != CANONICAL_IMAGE_ARCHIVE:
            raise ArtifactCacheSafetyError("canonical Flash-Next OCI path changed")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        parent_fd = os.open("/", flags)
        current = PurePosixPath("/")
        try:
            for part in PurePosixPath(str(archive.parent)).parts[1:]:
                current /= part
                child = os.open(part, flags, dir_fd=parent_fd)
                metadata = os.fstat(child)
                unsafe = (
                    metadata.st_uid != os.geteuid() or bool(metadata.st_mode & 0o022)
                    if current.is_relative_to(PurePosixPath("/home/aday"))
                    else bool(metadata.st_mode & 0o002)
                )
                if not stat.S_ISDIR(metadata.st_mode) or unsafe:
                    os.close(child)
                    raise ArtifactCacheSafetyError(
                        "canonical Flash-Next OCI directory is unsafe"
                    )
                os.close(parent_fd)
                parent_fd = child
            archive_fd = os.open(
                archive.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                before = os.fstat(archive_fd)
                _validate_flash_oci_layout_fd(
                    archive_fd,
                    manifest_digest=service.SGLANG_IMAGE_DIGEST.removeprefix(
                        "sha256:"
                    ),
                    config_digest=service.SGLANG_IMAGE_CONFIG_DIGEST.removeprefix(
                        "sha256:"
                    ),
                    archive_size=IMAGE_ARCHIVE_SIZE_BYTES,
                    expected_labels=service.runtime_contract.EXPECTED_IMAGE_LABELS,
                    allowed_link_counts=frozenset({1}),
                )
                payload_sha256 = self._fd_sha256(
                    archive_fd,
                    progress=progress,
                    total=descriptor.transfer_bytes_max,
                    expected_link_count=1,
                )
                after = os.fstat(archive_fd)
                path_metadata = os.stat(
                    archive.name, dir_fd=parent_fd, follow_symlinks=False
                )
            finally:
                os.close(archive_fd)
        finally:
            os.close(parent_fd)
        if (
            payload_sha256 != service.SGLANG_IMAGE_ARCHIVE_SHA256
            or (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            or (path_metadata.st_dev, path_metadata.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise ArtifactCacheSafetyError(
                "canonical Flash-Next OCI archive identity changed"
            )
        return archive, payload_sha256

    def _stage_oci(
        self,
        host: str,
        descriptor: ArtifactDescriptor,
        temporary: Path,
        filesystem_id: str,
        *,
        max_bytes_per_second: int,
        progress: Callable[[int, int], None],
    ) -> None:
        total = descriptor.transfer_bytes_max

        def canonical_keepalive(_completed: int, _total: int) -> None:
            progress(0, total)

        archive, payload_sha256 = self._canonical_flash_archive(
            descriptor, progress=canonical_keepalive
        )
        self._prepare_remote_temporary(
            host,
            temporary,
            filesystem_id,
            descriptor,
            directory=False,
        )
        image = self._remote_image_inspection(host, service.SGLANG_IMAGE_ID)
        if image is None:
            transferred = 0

            def transfer_progress(completed: int, _reported_total: int) -> None:
                nonlocal transferred
                transferred = max(transferred, min(total, completed))
                progress(transferred, total)

            def post_transfer_keepalive(
                _completed: int, _reported_total: int
            ) -> None:
                progress(transferred, total)

            transport = " ".join(self._ssh(host)[:-1])
            self._run_with_progress(
                [
                    str(cache_backend.HOST_BASH),
                    str(cache_backend.FLEET_LOW_PRIORITY),
                    "/usr/bin/rsync",
                    "-a",
                    "--inplace",
                    "--checksum",
                    "--protect-args",
                    f"--bwlimit={max(1, max_bytes_per_second // 1024)}",
                    "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                    "-e",
                    transport,
                    "--",
                    str(archive),
                    f"aday@{network_address(host)}:{temporary}",
                ],
                progress=transfer_progress,
                total=total,
                progress_probe=lambda: self._remote_staging_usage(
                    host, temporary, filesystem_id, descriptor
                ),
            )
            remote_sha = self._remote_sha256(
                host,
                temporary,
                progress=post_transfer_keepalive,
                total=total,
            )
            if remote_sha != payload_sha256:
                raise ArtifactCacheSafetyError(
                    "worker Flash-Next OCI archive transfer changed"
                )
            self._run_with_progress(
                [
                    *self._ssh(host),
                    shlex.join(
                        [
                            "/usr/bin/env",
                            "-i",
                            "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
                            "HOME=/home/aday",
                            "LANG=C",
                            "LC_ALL=C",
                            "/usr/bin/bash",
                            cache_backend.REMOTE_WRAPPER,
                            "/usr/bin/bash",
                            cache_backend.REMOTE_DOCKER,
                            "image",
                            "load",
                            "--input",
                            str(temporary),
                        ]
                    ),
                ],
                progress=post_transfer_keepalive,
                total=total,
                timeout=3600,
            )
            image = self._remote_image_inspection(host, service.SGLANG_IMAGE_ID)
            if image is None:
                raise ArtifactCacheSafetyError(
                    "loaded Flash-Next image identity is absent"
                )
        receipt = {
            "schema_version": cache_backend.OCI_RECEIPT_SCHEMA,
            "image_id": service.SGLANG_IMAGE_ID,
            "image_size_bytes": int(image["Size"]),
            "archive_payload_sha256": payload_sha256,
        }
        self._commit_remote_oci_receipt(
            host, temporary, filesystem_id, descriptor, receipt
        )
        progress(total, total)

    @staticmethod
    def _validate_descriptor(descriptor: ArtifactDescriptor) -> None:
        try:
            binding = service.load_production_binding(verify_release_hashes=False)
            expected_raw = {
                item["artifact_id"]: item
                for item in promoted_artifact_cache(binding)["artifacts"]
            }
            expected = ArtifactDescriptor.from_dict(
                expected_raw[descriptor.artifact_id]
            )
        except (
            KeyError,
            OSError,
            ValueError,
            service.FlashNextServiceError,
            RemoteFlashNextServiceError,
        ) as exc:
            raise ArtifactCacheSafetyError(
                "qualified Flash-Next cache descriptor is unavailable"
            ) from exc
        if descriptor != expected:
            raise ArtifactCacheSafetyError("Flash-Next cache descriptor changed")

    @staticmethod
    def _verify_canonical_tree(
        descriptor: ArtifactDescriptor,
        progress_check: Callable[[], None],
    ) -> None:
        FlashNextArtifactBackend._validate_descriptor(descriptor)
        progress_check()
        binding = service.load_production_binding(verify_release_hashes=True)
        progress_check()
        if descriptor.artifact_id == RELEASE_ARTIFACT_ID:
            expected_path = service.RELEASE_DIR
            expected_digest = binding.release_tree_sha256
        elif descriptor.artifact_id == MODEL_ARTIFACT_ID:
            expected_path = MATERIALIZED_MODEL_DIR
            expected_digest = str(
                getattr(binding, "materialized_checkpoint_tree_sha256", "")
            )
        else:
            raise ArtifactCacheSafetyError("Flash-Next tree artifact changed")
        if (
            Path(descriptor.canonical_path) != expected_path
            or descriptor.digest_sha256 != expected_digest
            or descriptor.digest_sha256 != _sha256(expected_path / "SHA256SUMS")
        ):
            raise ArtifactCacheSafetyError("canonical Flash-Next tree changed")

    def _stage_tree(
        self,
        host: str,
        descriptor: ArtifactDescriptor,
        temporary: Path,
        filesystem_id: str,
        *,
        max_bytes_per_second: int,
        progress: Callable[[int, int], None],
    ) -> None:
        self._verify_canonical_tree(
            descriptor, lambda: progress(0, descriptor.transfer_bytes_max)
        )
        files = self._manifest_files(descriptor)
        self._prepare_remote_temporary(
            host, temporary, filesystem_id, descriptor, directory=True
        )
        transport = " ".join(self._ssh(host)[:-1])
        command = [
            str(cache_backend.HOST_BASH),
            str(cache_backend.FLEET_LOW_PRIORITY),
            "/usr/bin/rsync",
            "-aH",
            "--checksum",
            "--protect-args",
            f"--bwlimit={max(1, max_bytes_per_second // 1024)}",
            "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
            "-e",
            transport,
            "--",
            *files,
            f"aday@{network_address(host)}:{temporary}/",
        ]
        self._run_with_progress(
            command,
            cwd=descriptor.canonical_path,
            progress=progress,
            total=descriptor.transfer_bytes_max,
            progress_probe=lambda: self._remote_staging_usage(
                host, temporary, filesystem_id, descriptor
            ),
        )
        progress(descriptor.transfer_bytes_max, descriptor.transfer_bytes_max)


def _ssh() -> list[str]:
    return FlashNextArtifactBackend._ssh(HOST)


def _remote_python(script: str, *arguments: str, timeout: float = 120) -> dict[str, Any]:
    command = [
        *_ssh(),
        shlex.join(
            [
                "/usr/bin/env",
                "-i",
                "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME=/home/aday",
                "LANG=C",
                "LC_ALL=C",
                "/usr/bin/bash",
                REMOTE_WRAPPER,
                REMOTE_PYTHON,
                "-I",
                "-S",
                "-B",
                "-c",
                script,
                HOSTNAME,
                *arguments,
            ]
        ),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RemoteFlashNextTransportError("remote worker transport failed") from exc
    if result.returncode != 0:
        raise RemoteFlashNextTransportError("remote worker proof command failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteFlashNextTransportError("remote worker proof is malformed") from exc
    if not isinstance(value, dict):
        raise RemoteFlashNextTransportError("remote worker proof is not an object")
    return value


def _write_private(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RemoteFlashNextServiceError("private request path already exists")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RemoteFlashNextServiceError("private request write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private(path: Path, payload: bytes) -> None:
    """Create one private file, or accept only its byte-identical prior write."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        _write_private(path, payload)
        return
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or path.read_bytes() != payload
    ):
        raise RemoteFlashNextServiceError("private request retry identity changed")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_private(temporary, payload)
    os.replace(temporary, path)


def _private_json(path: Path, *, maximum: int = 64 * 1024) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise RemoteFlashNextServiceError("private tunnel/request receipt is unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RemoteFlashNextServiceError("private tunnel/request receipt is malformed")
    return value


def _prepare_remote_dirs(run_dir: str) -> dict[str, Any]:
    script = r'''
import json,os,pathlib,stat,sys
expected,raw=sys.argv[1:3]
assert os.uname().nodename==expected
root=pathlib.Path("/home/aday/.local/state/fleet-compute/runs")
run=pathlib.Path(raw); assert run.parent==root and run.name.startswith("fr-")
root.mkdir(mode=0o700,parents=True,exist_ok=True)
for path in (root,run,run/"source"):
 path.mkdir(mode=0o700,exist_ok=True); path.chmod(0o700)
 meta=path.lstat(); assert stat.S_ISDIR(meta.st_mode) and meta.st_uid==os.geteuid() and not meta.st_mode&0o077
assert set(run.iterdir()) <= {run/"source",run/"flash-next-remote-request.json",run/"flash-next-remote-container.json"}
assert set((run/"source").iterdir()) <= {run/"source"/"qwen_flash_next_remote_service_worker.py"}
for item in [entry for entry in run.iterdir() if entry != run/"source"] + list((run/"source").iterdir()):
 meta=item.lstat(); assert stat.S_ISREG(meta.st_mode) and meta.st_uid==os.geteuid() and meta.st_nlink==1 and not meta.st_mode&0o077
v=os.statvfs(run)
print(json.dumps({"filesystem_id":str(run.lstat().st_dev),"free_bytes":v.f_bavail*v.f_frsize,"free_inodes":v.f_favail},sort_keys=True))
'''
    return _remote_python(script, run_dir)


def _remote_run_absent(run_dir: str) -> bool:
    script = r'''
import json,os,pathlib,sys
expected,raw=sys.argv[1:3]
assert os.uname().nodename==expected
root=pathlib.PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
run=pathlib.PurePosixPath(raw)
assert run.parent==root and run.name.startswith("fr-") and ".." not in run.parts
try: pathlib.Path(raw).lstat(); absent=False
except FileNotFoundError: absent=True
print(json.dumps({"absent":absent},sort_keys=True))
'''
    return _remote_python(script, run_dir).get("absent") is True


def _remote_named_container_absent(name: str) -> bool:
    script = r'''
import json,re,subprocess,sys
expected,name=sys.argv[1:3]
import os
assert os.uname().nodename==expected
assert re.fullmatch(r"aeon-qwen38-flash-next-179-fr-[0-9a-f]{32}",name)
result=subprocess.run(["/home/aday/bin/fleet-low-priority","/home/aday/bin/docker",
 "container","inspect",name],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=30,
 env={"HOME":"/home/aday","PATH":"/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin","LANG":"C","LC_ALL":"C"})
absent=(result.returncode==1 and re.search(r"(?:No such object|No such container):\\s*"+re.escape(name)+r"(?:\\s|$)",result.stderr) is not None)
assert result.returncode==0 or absent
print(json.dumps({"absent":absent},sort_keys=True))
'''
    return _remote_python(script, name).get("absent") is True


def _stage_file(source: Path, destination: str, *, timeout: float = 600) -> None:
    transport = " ".join(_ssh()[:-1])
    result = subprocess.run(
        [
            LOW_PRIORITY,
            "/usr/bin/rsync",
            "-aH",
            "--checksum",
            "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
            "--protect-args",
            "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
            "-e",
            transport,
            "--",
            str(source),
            f"aday@{network_address(HOST)}:{destination}",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RemoteFlashNextTransportError("remote service source/request staging failed")


def _remote_metrics(run_dir: str) -> dict[str, Any]:
    script = r'''
import json,os,pathlib,stat,sys
expected,raw=sys.argv[1:3]
assert os.uname().nodename==expected
root=pathlib.Path("/home/aday/.local/state/fleet-compute/runs")
run=pathlib.Path(raw); assert run.parent==root
meta=run.lstat(); assert stat.S_ISDIR(meta.st_mode) and meta.st_uid==os.geteuid() and not meta.st_mode&0o077
total=0
for item in run.rglob("*"):
 m=item.lstat(); assert m.st_uid==os.geteuid() and not stat.S_ISLNK(m.st_mode)
 if stat.S_ISREG(m.st_mode): total += m.st_blocks*512
 elif not stat.S_ISDIR(m.st_mode): raise AssertionError
v=os.statvfs(run)
print(json.dumps({"filesystem_id":str(meta.st_dev),"free_bytes":v.f_bavail*v.f_frsize,"free_inodes":v.f_favail,"allocated_bytes":total},sort_keys=True))
'''
    return _remote_python(script, run_dir)


def _remote_action(
    action: str, run_dir: str, request_sha256: str, *, timeout: float = 120
) -> dict[str, Any]:
    source = f"{run_dir}/source/{worker.WORKER_NAME}"
    request = f"{run_dir}/{REQUEST_NAME}"
    command = [
        *_ssh(),
        shlex.join(
            [
                "/usr/bin/env",
                "-i",
                "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME=/home/aday",
                "LANG=C",
                "LC_ALL=C",
                "/usr/bin/bash",
                REMOTE_WRAPPER,
                REMOTE_PYTHON,
                "-I",
                "-S",
                "-B",
                source,
                action,
                request,
                request_sha256,
            ]
        ),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RemoteFlashNextTransportError("remote service lifecycle transport failed") from exc
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteFlashNextTransportError("remote service lifecycle response is malformed") from exc
    if not isinstance(value, dict):
        raise RemoteFlashNextTransportError("remote service lifecycle response is not an object")
    if result.returncode != 0 or value.get("ok") is not True:
        raise RemoteFlashNextServiceError(
            str(value.get("detail") or value.get("error") or "remote lifecycle refused")[:500]
        )
    return value


def _ports(physical_gpu: Any) -> tuple[int, int]:
    if type(physical_gpu) is not int or physical_gpu not in {0, 1}:
        raise RemoteFlashNextServiceError("saved remote physical-GPU slot is malformed")
    return LOCAL_PORT_BASE + physical_gpu, REMOTE_PORT_BASE + physical_gpu


def _process_start_ticks(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = payload.rfind(")")
    if end < 0:
        raise RemoteFlashNextServiceError("tunnel process stat is malformed")
    return int(payload[end + 2 :].split()[19])


def _process_argv(pid: int) -> list[str]:
    metadata = Path(f"/proc/{pid}").stat()
    if metadata.st_uid != os.geteuid():
        raise RemoteFlashNextServiceError("tunnel process owner changed")
    raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if raw and raw[-1] == b"":
        raw.pop()
    return [item.decode("utf-8") for item in raw]


def _tunnel_argv(local_port: int, remote_port: int) -> list[str]:
    base = _ssh()
    return [
        *base[:-1],
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        base[-1],
    ]


def _pid_absent(pid: Any) -> bool:
    if type(pid) is not int or pid <= 1:
        return False
    try:
        Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _tunnel_exact(receipt: Mapping[str, Any]) -> bool:
    pid = receipt.get("pid")
    ticks = receipt.get("start_ticks")
    try:
        return (
            type(pid) is int
            and pid > 1
            and type(ticks) is int
            and _process_start_ticks(pid) == ticks
            and _process_argv(pid)
            == _tunnel_argv(int(receipt["local_port"]), int(receipt["remote_port"]))
        )
    except (OSError, ValueError, KeyError, UnicodeDecodeError, RemoteFlashNextServiceError):
        return False


def _bounded_body(response: requests.Response, maximum: int) -> bytes:
    payload = bytearray()
    try:
        advertised = response.headers.get("content-length")
        if advertised is not None and int(advertised) > maximum:
            raise RemoteFlashNextServiceError("loopback response exceeded its bound")
        for chunk in response.iter_content(chunk_size=min(64 * 1024, maximum + 1)):
            payload.extend(chunk)
            if len(payload) > maximum:
                raise RemoteFlashNextServiceError("loopback response exceeded its bound")
    finally:
        response.close()
    return bytes(payload)


def _endpoint_ready(local_port: int, *, semantic: bool) -> bool:
    options = {
        "timeout": (2, 20),
        "allow_redirects": False,
        "proxies": {"http": "", "https": ""},
        "stream": True,
    }
    base = f"http://127.0.0.1:{local_port}"
    try:
        health = requests.get(f"{base}/health", **options)
        health_status = health.status_code
        _bounded_body(health, 64 * 1024)
        models = requests.get(f"{base}/v1/models", **options)
        models_status = models.status_code
        model_payload = json.loads(_bounded_body(models, 256 * 1024))
        identities = {
            item.get("id")
            for item in model_payload.get("data", [])
            if isinstance(item, Mapping)
        }
        if health_status != 200 or models_status != 200 or identities != {service.SERVED_ALIAS}:
            return False
        if not semantic:
            return True
        response = requests.post(
            f"{base}/v1/chat/completions",
            json={
                "model": service.SERVED_ALIAS,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly AEON_READY and nothing else.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
            },
            **options,
        )
        status = response.status_code
        value = json.loads(_bounded_body(response, 256 * 1024))
        choices = value.get("choices") if isinstance(value, Mapping) else None
        if status != 200 or value.get("model") != service.SERVED_ALIAS or not isinstance(choices, list):
            return False
        text = " ".join(
            str((choice.get("message") or {}).get(field) or "")
            for choice in choices
            if isinstance(choice, Mapping)
            for field in ("content", "reasoning_content")
        )
        return "AEON_READY" in text
    except (
        requests.RequestException,
        RemoteFlashNextServiceError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _start_tunnel(
    run_dir: Path,
    runtime_id: str,
    request_sha256: str,
    physical_gpu: int,
) -> tuple[int, int]:
    local_port, remote_port = _ports(physical_gpu)
    receipt_path = run_dir / TUNNEL_RECEIPT
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _private_json(receipt_path)
        if (
            receipt.get("runtime_id") == runtime_id
            and receipt.get("request_sha256") == request_sha256
            and receipt.get("state") == "active"
            and receipt.get("local_port") == local_port
            and receipt.get("remote_port") == remote_port
            and _tunnel_exact(receipt)
            and _endpoint_ready(local_port, semantic=True)
        ):
            return int(receipt["pid"]), int(receipt["start_ticks"])
        raise RemoteFlashNextServiceError("remote service tunnel receipt already exists")
    intent = {
        "schema_version": 1,
        "runtime_id": runtime_id,
        "request_sha256": request_sha256,
        "state": "starting",
        "local_port": local_port,
        "remote_port": remote_port,
        "pid": None,
        "start_ticks": None,
    }
    _atomic_json(receipt_path, intent)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", local_port))
    except OSError as exc:
        raise RemoteFlashNextServiceError("remote service loopback port is unavailable") from exc
    finally:
        probe.close()
    process = subprocess.Popen(
        _tunnel_argv(local_port, remote_port),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    receipt = {
        **intent,
        "state": "active",
        "pid": process.pid,
        "start_ticks": _process_start_ticks(process.pid),
    }
    _atomic_json(receipt_path, receipt)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RemoteFlashNextServiceError("remote service tunnel exited before readiness")
        if not _tunnel_exact(receipt):
            raise RemoteFlashNextServiceError("remote service tunnel identity changed")
        if _endpoint_ready(local_port, semantic=True):
            return process.pid, int(receipt["start_ticks"])
        time.sleep(2)
    raise RemoteFlashNextServiceError("remote service semantic tunnel readiness timed out")


def _stop_tunnel(run_dir: Path, runtime_id: str, request_sha256: str) -> bool:
    path = run_dir / TUNNEL_RECEIPT
    try:
        receipt = _private_json(path)
    except FileNotFoundError:
        return False
    if receipt.get("runtime_id") != runtime_id or receipt.get("request_sha256") != request_sha256:
        return False
    if receipt.get("state") == "stopped":
        return _pid_absent(receipt.get("pid"))
    if not _tunnel_exact(receipt):
        if _pid_absent(receipt.get("pid")):
            _atomic_json(path, {**receipt, "state": "stopped"})
            return True
        return False
    pid = int(receipt["pid"])
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if _pid_absent(pid):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            _atomic_json(path, {**receipt, "state": "stopped"})
            return True
        if not _tunnel_exact(receipt):
            return False
        time.sleep(0.1)
    return False


def _binding_runtime_environment(binding: service.ProductionBinding) -> dict[str, str]:
    bound_environment = dict(binding.runtime_environment)
    if bound_environment != RUNTIME_ENVIRONMENT:
        raise RemoteFlashNextServiceError(
            "production binding runtime environment changed"
        )
    runtime = service._safe_json(binding.release_dir / "RUNTIME_CONFIG.json")
    arm = runtime.get("arms", {}).get("tuned_mtp_on_winner", {})
    mtp = arm.get("runtime_config") if isinstance(arm, Mapping) else None
    if (
        not isinstance(mtp, Mapping)
        or arm.get("environment") != bound_environment
        or mtp.get("runtime_environment") != bound_environment
        or mtp.get("ragged_verify_mode") != "static"
    ):
        raise RemoteFlashNextServiceError(
            "qualified winner does not bind its exact static verification environment"
        )
    return bound_environment


def _cache_binding(binding: ArtifactCacheBinding) -> dict[str, Any]:
    return {
        "artifact_id": binding.artifact_id,
        "worker_path": binding.worker_path,
        "digest_sha256": binding.digest_sha256,
        "filesystem_id": binding.filesystem_id,
        "size_bytes": binding.size_bytes,
        "inode_count": binding.inode_count,
        **(
            {"payload_sha256": binding.payload_sha256}
            if binding.artifact_id == IMAGE_ARTIFACT_ID
            else {}
        ),
    }


def _request_payload(
    context: RuntimeContext,
    binding: service.ProductionBinding,
) -> dict[str, Any]:
    release = context.cached_artifacts[RELEASE_ARTIFACT_ID]
    model_binding = context.cached_artifacts[MODEL_ARTIFACT_ID]
    image = context.cached_artifacts[IMAGE_ARTIFACT_ID]
    _local_port, remote_port = _ports(context.lease.physical_gpu)
    bound_environment = _binding_runtime_environment(binding)
    if any(
        key in context.lease.required_environment
        and context.lease.required_environment[key] != value
        for key, value in bound_environment.items()
    ):
        raise RemoteFlashNextServiceError(
            "lease environment conflicts with qualified runtime"
        )
    environment = {
        **bound_environment,
        **context.lease.required_environment,
    }
    return {
        "schema_version": worker.SCHEMA,
        "contract_sha256": remote_staging_contract_sha256(),
        "runtime_id": context.runtime_id,
        "profile_id": PROFILE_ID,
        "host": HOST,
        "hostname": HOSTNAME,
        "run_dir": str(context.scratch_path),
        "source_path": f"{context.scratch_path}/source/{worker.WORKER_NAME}",
        "source_sha256": worker_source_sha256(),
        "binding_sha256": binding.binding_sha256,
        "release_manifest_sha256": binding.release_manifest_sha256,
        "release_tree_sha256": binding.release_tree_sha256,
        "checkpoint_tree_sha256": binding.checkpoint_tree_sha256,
        "materialized_checkpoint_tree_sha256": (
            binding.materialized_checkpoint_tree_sha256
        ),
        "ple_materialization_manifest_sha256": (
            binding.ple_materialization_manifest_sha256
        ),
        "ple_materializer_sha256": binding.ple_materializer_sha256,
        "materialization_receipt_sha256": (
            binding.materialization_receipt_sha256
        ),
        "runtime_config_sha256": binding.runtime_config_sha256,
        "qualification_sha256": binding.qualification_sha256,
        "qualification_mtp_off_sha256": binding.qualification_mtp_off_sha256,
        "qualification_mtp_on_sha256": binding.qualification_mtp_on_sha256,
        "release": _cache_binding(release),
        "materialized_model": _cache_binding(model_binding),
        "image": _cache_binding(image),
        "lease": {
            "claim_id": context.lease.claim_id,
            "owner": context.lease.owner,
            "physical_gpu": context.lease.physical_gpu,
            "gpu_uuid": context.lease.gpu_uuid,
            "vram_budget_gb": context.lease.vram_budget_gb,
            "exclusive": context.lease.exclusive,
            "model": context.lease.model,
            "memory_total_mib": context.lease.memory_total_mib,
        },
        "container": {
            "name": f"aeon-qwen38-flash-next-179-{context.runtime_id}",
            "host_port": remote_port,
            "container_port": CONTAINER_PORT,
            "image_reference": service.SGLANG_IMAGE_REFERENCE,
            "image_id": service.SGLANG_IMAGE_ID,
            "task_memory_bytes": binding.task_memory_bytes,
            "shm_bytes": 32 * 1024**3,
            "command": list(binding.container_command),
            "command_sha256": binding.command_sha256,
            "environment": environment,
            "environment_sha256": _canonical_sha(environment),
        },
    }


def _pidless_request(
    runtime: Mapping[str, Any],
    *,
    allowed_states: frozenset[str] = frozenset({"starting", "quarantined"}),
) -> tuple[dict[str, Any], str, service.ProductionBinding]:
    """Rebind an unpublished durable runtime to its exact private request."""

    runtime_id = str(runtime.get("runtime_id") or "")
    run_dir = Path(str(runtime.get("run_dir") or ""))
    if (
        _RUNTIME.fullmatch(runtime_id) is None
        or runtime.get("profile_id") != PROFILE_ID
        or runtime.get("adapter") != ADAPTER_NAME
        or runtime.get("mode") != "service"
        or runtime.get("state") not in allowed_states
        or runtime.get("host") != HOST
        or run_dir != Path(RUN_ROOT) / runtime_id
        or runtime.get("job_id") is not None
        or runtime.get("payload_json") not in {None, "{}"}
        or runtime.get("pid") is not None
        or runtime.get("process_identity") is not None
        or runtime.get("endpoint") is not None
        or runtime.get("physical_gpu") not in {0, 1}
        or runtime.get("vram_budget_gb") != VRAM_BUDGET_GB
        or runtime.get("exclusive") not in {True, 1}
    ):
        raise RemoteFlashNextServiceError(
            "PID-less remote runtime is outside its durable recovery contract"
        )
    request_path = run_dir / REQUEST_NAME
    request = _private_json(request_path, maximum=worker.MAX_JSON_BYTES)
    canonical = (
        json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if request_path.read_bytes() != canonical:
        raise RemoteFlashNextServiceError("private recovery request is not canonical")
    request_sha256 = hashlib.sha256(canonical).hexdigest()
    binding = service.load_production_binding(verify_release_hashes=False)
    expected_binding = {
        "binding_sha256": binding.binding_sha256,
        "release_manifest_sha256": binding.release_manifest_sha256,
        "release_tree_sha256": binding.release_tree_sha256,
        "checkpoint_tree_sha256": binding.checkpoint_tree_sha256,
        "materialized_checkpoint_tree_sha256": (
            binding.materialized_checkpoint_tree_sha256
        ),
        "ple_materialization_manifest_sha256": (
            binding.ple_materialization_manifest_sha256
        ),
        "ple_materializer_sha256": binding.ple_materializer_sha256,
        "materialization_receipt_sha256": (
            binding.materialization_receipt_sha256
        ),
        "runtime_config_sha256": binding.runtime_config_sha256,
        "qualification_sha256": binding.qualification_sha256,
        "qualification_mtp_off_sha256": binding.qualification_mtp_off_sha256,
        "qualification_mtp_on_sha256": binding.qualification_mtp_on_sha256,
    }
    lease = request.get("lease")
    if (
        request.get("schema_version") != worker.SCHEMA
        or request.get("contract_sha256") != remote_staging_contract_sha256()
        or request.get("runtime_id") != runtime_id
        or request.get("profile_id") != PROFILE_ID
        or request.get("host") != HOST
        or request.get("hostname") != HOSTNAME
        or request.get("run_dir") != str(run_dir)
        or request.get("source_path")
        != f"{run_dir}/source/{worker.WORKER_NAME}"
        or request.get("source_sha256") != worker_source_sha256()
        or any(request.get(key) != value for key, value in expected_binding.items())
        or not isinstance(lease, Mapping)
        or lease.get("claim_id") != runtime.get("claim_id")
        or lease.get("owner") != runtime.get("owner")
        or lease.get("physical_gpu") != runtime.get("physical_gpu")
        or lease.get("gpu_uuid") != runtime.get("gpu_uuid")
        or lease.get("vram_budget_gb") != runtime.get("vram_budget_gb")
        or lease.get("exclusive") is not True
    ):
        raise RemoteFlashNextServiceError("private recovery request identity changed")
    try:
        worker._validate_lease(request)
        worker._validate_container_request(request)
    except worker.RemoteWorkerError as exc:
        raise RemoteFlashNextServiceError(
            "private recovery request is outside the worker contract"
        ) from exc
    cache_expected = {
        "release": (RELEASE_ARTIFACT_ID, binding.release_tree_sha256),
        "materialized_model": (
            MODEL_ARTIFACT_ID,
            binding.materialized_checkpoint_tree_sha256,
        ),
        "image": (
            IMAGE_ARTIFACT_ID,
            service.SGLANG_IMAGE_ID.removeprefix("sha256:"),
        ),
    }
    for field, (artifact_id, digest) in cache_expected.items():
        item = request.get(field)
        if (
            not isinstance(item, Mapping)
            or item.get("artifact_id") != artifact_id
            or item.get("digest_sha256") != digest
        ):
            raise RemoteFlashNextServiceError(
                "private recovery cache binding changed"
            )
        try:
            PurePosixPath(str(item.get("worker_path"))).relative_to(
                PurePosixPath(str(CACHE_ROOT))
            )
        except ValueError as exc:
            raise RemoteFlashNextServiceError(
                "private recovery cache path escaped Fleet"
            ) from exc
    return request, request_sha256, binding


def _prelaunch_tunnel_absent(
    run_dir: Path, runtime_id: str, request_sha256: str, physical_gpu: int
) -> bool:
    path = run_dir / TUNNEL_RECEIPT
    try:
        receipt = _private_json(path)
    except FileNotFoundError:
        # The launch path durably writes a private intent before Popen.  With the
        # current source/contract hash already rebound above, no receipt means
        # this adapter never created a tunnel for the request.
        return True
    local_port, remote_port = _ports(physical_gpu)
    if (
        receipt.get("runtime_id") != runtime_id
        or receipt.get("request_sha256") != request_sha256
        or receipt.get("local_port") != local_port
        or receipt.get("remote_port") != remote_port
    ):
        return False
    if receipt.get("state") == "stopped":
        return _pid_absent(receipt.get("pid"))
    if receipt.get("state") == "active":
        return _stop_tunnel(run_dir, runtime_id, request_sha256)
    # A crash between Popen and PID receipt publication is intentionally
    # ambiguous.  Never scan/adopt or signal a process without that receipt.
    return False


def _cleanup_marker(
    runtime_id: str,
    request_sha256: str,
    binding_sha256: str,
    *,
    state: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runtime_id": runtime_id,
        "request_sha256": request_sha256,
        "binding_sha256": binding_sha256,
        "contract_sha256": remote_staging_contract_sha256(),
        "state": state,
    }


def _recover_pidless_runtime(
    runtime: Mapping[str, Any],
    *,
    allowed_states: frozenset[str] = frozenset({"starting", "quarantined"}),
) -> bool:
    request, request_sha256, binding = _pidless_request(
        runtime, allowed_states=allowed_states
    )
    runtime_id = str(runtime["runtime_id"])
    run_dir = Path(str(runtime["run_dir"]))
    marker_path = run_dir / PRELAUNCH_CLEANUP_RECEIPT
    expected_cleaning = _cleanup_marker(
        runtime_id, request_sha256, binding.binding_sha256, state="cleaning"
    )
    expected_complete = {
        **expected_cleaning,
        "state": "complete",
    }
    try:
        marker = _private_json(marker_path)
    except FileNotFoundError:
        _atomic_json(marker_path, expected_cleaning)
        marker = expected_cleaning
    if marker not in (expected_cleaning, expected_complete):
        raise RemoteFlashNextServiceError(
            "prelaunch cleanup marker identity changed"
        )
    if not _prelaunch_tunnel_absent(
        run_dir,
        runtime_id,
        request_sha256,
        int(runtime["physical_gpu"]),
    ):
        raise RemoteFlashNextServiceError(
            "prelaunch tunnel absence is not exactly proven"
        )
    container_name = str(request["container"]["name"])
    if _remote_run_absent(str(run_dir)):
        if not _remote_named_container_absent(container_name):
            raise RemoteFlashNextServiceError(
                "remote scratch is absent but the exact container remains"
            )
    else:
        _prepare_remote_dirs(str(run_dir))
        _stage_file(WORKER_SOURCE, f"{run_dir}/source/{worker.WORKER_NAME}")
        _stage_file(run_dir / REQUEST_NAME, f"{run_dir}/{REQUEST_NAME}")
        result = _remote_action(
            "cleanup", str(run_dir), request_sha256, timeout=300
        )
        if (
            result.get("state") != "cleaned"
            or result.get("process_absent") is not True
            or result.get("cache_entries_removed") != 0
            or result.get("docker_images_removed") != 0
            or not _remote_run_absent(str(run_dir))
            or not _remote_named_container_absent(container_name)
        ):
            raise RemoteFlashNextServiceError(
                "prelaunch remote cleanup proof changed"
            )
    if marker != expected_complete:
        _atomic_json(marker_path, expected_complete)
    return True


class _PrepareHeartbeat:
    def __init__(self, context: RuntimeContext) -> None:
        self.context = context
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PrepareHeartbeat":
        self.context.heartbeat(None, "Validating/staging exact remote Flash-Next service")
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise RemoteFlashNextServiceError("remote preparation heartbeat failed") from self.error

    def _run(self) -> None:
        while not self.stop.wait(60):
            try:
                self.context.heartbeat(None, "Remote Flash-Next validation/staging remains active")
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenFlashNextRemoteServiceAdapter:
    """Serve the exact qualified Flash-Next release on either leased .179 GPU."""

    def __init__(self) -> None:
        self.artifact_cache_backend = FlashNextArtifactBackend()
        self._prepared: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_context(
        context: RuntimeContext, binding: service.ProductionBinding
    ) -> None:
        profile = context.profile
        lease = context.lease
        expected_cache = promoted_artifact_cache(binding)
        cold_peak = validate_promoted_artifact_cache(
            expected_cache, remote_artifact_identity(binding)
        )
        expected_min_disk_gb = (
            cold_peak
            + RUNTIME_GROWTH_BYTES_MAX
            + WORKER_FREE_RESERVE_BYTES
            + 999_999_999
        ) // 1_000_000_000
        task_memory_gib = binding.task_memory_bytes / 1024**3
        if (
            profile.profile_id != PROFILE_ID
            or profile.enabled is not True
            or profile.adapter != ADAPTER_NAME
            or profile.service_id != service.SERVICE_ID
            or profile.variant_priority != 10
            or profile.max_replicas != 2
            or profile.personal_priority != 30
            or profile.vram_budget_gb != VRAM_BUDGET_GB
            or profile.min_physical_vram_gb != MIN_PHYSICAL_VRAM_GB
            or profile.exclusive is not True
            or profile.min_host_memory_gb < task_memory_gib
            or profile.min_host_commit_gb < task_memory_gib
            or profile.min_shm_free_gb != 32
            or profile.stage_bytes_max != cold_peak
            or profile.runtime_growth_bytes_max != RUNTIME_GROWTH_BYTES_MAX
            or profile.worker_free_reserve_bytes != WORKER_FREE_RESERVE_BYTES
            or profile.min_disk_free_gb != expected_min_disk_gb
            or profile.serving_pool_id != service.SERVING_POOL_ID
            or profile.lane_max_replicas != service.FLASH_LANE_MAX_REPLICAS
            or profile.artifact_identity != remote_artifact_identity(binding)
            or profile.artifact_cache is None
            or profile.artifact_cache.to_dict() != expected_cache
        ):
            raise RemoteFlashNextServiceError("remote Fleet profile is not fully promoted")
        if any(value == service.ZERO_SHA256 for value in profile.artifact_identity.values()):
            raise RemoteFlashNextServiceError("remote Fleet profile retains placeholders")
        placements = [item for item in profile.placements if item.enabled]
        model = str(lease.model or "").casefold()
        if (
            len(placements) != 1
            or placements[0].host != HOST
            or placements[0].physical_gpu is not None
            or lease.host != HOST
            or lease.physical_gpu not in {0, 1}
            or lease.memory_total_mib is None
            or lease.memory_total_mib < int(MIN_PHYSICAL_VRAM_GB * 1024)
            or lease.memory_total_mib / 1024 - VRAM_BUDGET_GB < 6
            or "rtx pro 6000" not in model
            or "blackwell" not in model
            or abs(lease.vram_budget_gb - VRAM_BUDGET_GB) > 1e-9
            or lease.exclusive is not True
            or context.job_id is not None
            or context.payload
            or context.scratch_path != lease.run_dir
            or PurePosixPath(str(lease.run_dir)).parent != PurePosixPath(str(RUN_ROOT))
            or str(context.run_dir) != lease.run_dir
        ):
            raise RemoteFlashNextServiceError("lease is not an exact .179 PRO 6000 lease")
        if set(context.cached_artifacts) != {
            RELEASE_ARTIFACT_ID,
            MODEL_ARTIFACT_ID,
            IMAGE_ARTIFACT_ID,
        }:
            raise RemoteFlashNextServiceError("remote cache bundle is incomplete")
        release = context.cached_artifacts[RELEASE_ARTIFACT_ID]
        model_binding = context.cached_artifacts[MODEL_ARTIFACT_ID]
        image = context.cached_artifacts[IMAGE_ARTIFACT_ID]
        if (
            release.kind is not ArtifactKind.MANIFESTED_TREE
            or release.digest_sha256 != binding.release_tree_sha256
            or model_binding.kind is not ArtifactKind.MANIFESTED_TREE
            or model_binding.digest_sha256
            != binding.materialized_checkpoint_tree_sha256
            or image.kind is not ArtifactKind.OCI_ARCHIVE
            or image.digest_sha256
            != service.SGLANG_IMAGE_ID.removeprefix("sha256:")
            or image.payload_sha256 is None
        ):
            raise RemoteFlashNextServiceError("remote cache bundle identity changed")
        for descriptor in profile.artifact_cache.artifacts:
            FlashNextArtifactBackend._validate_descriptor(descriptor)

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if _RUNTIME.fullmatch(context.runtime_id) is None:
            raise RemoteFlashNextServiceError("remote service runtime ID is malformed")
        with _PrepareHeartbeat(context):
            binding = service.load_production_binding(verify_release_hashes=True)
            self._validate_context(context, binding)
            request = _request_payload(context, binding)
            request_bytes = (
                json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
            request_sha256 = hashlib.sha256(request_bytes).hexdigest()
            local_request = context.run_dir / REQUEST_NAME
            _ensure_private(local_request, request_bytes)
            before = _prepare_remote_dirs(str(context.scratch_path))
            _stage_file(
                WORKER_SOURCE,
                f"{context.scratch_path}/source/{worker.WORKER_NAME}",
            )
            _stage_file(local_request, f"{context.scratch_path}/{REQUEST_NAME}")
            context.startup_check()
            preflight = _remote_action(
                "preflight", str(context.scratch_path), request_sha256, timeout=7_200
            )
            expected = {
                "state": "verified",
                "request_sha256": request_sha256,
                "contract_sha256": remote_staging_contract_sha256(),
                "binding_sha256": binding.binding_sha256,
                "release_tree_sha256": binding.release_tree_sha256,
                "materialized_checkpoint_tree_sha256": (
                    binding.materialized_checkpoint_tree_sha256
                ),
                "materialization_receipt_sha256": (
                    binding.materialization_receipt_sha256
                ),
                "image_digest_sha256": service.SGLANG_IMAGE_ID.removeprefix(
                    "sha256:"
                ),
                "command_sha256": binding.command_sha256,
                "environment_sha256": request["container"]["environment_sha256"],
            }
            if any(preflight.get(key) != value for key, value in expected.items()):
                raise RemoteFlashNextServiceError("remote semantic preflight changed")
            after = _remote_metrics(str(context.scratch_path))
            if after.get("filesystem_id") != before.get("filesystem_id"):
                raise RemoteFlashNextServiceError("remote run filesystem changed during staging")
        with self._lock:
            self._prepared[context.runtime_id] = request_sha256
        return StoragePreparationResult(
            scratch_path=context.scratch_path,
            filesystem_id=str(after["filesystem_id"]),
            free_bytes_after_stage=int(after["free_bytes"]),
            free_inodes_after_stage=int(after["free_inodes"]),
            staged_bytes=int(after["allocated_bytes"]),
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            request_sha256 = self._prepared.get(context.runtime_id)
        if request_sha256 is None:
            raise AdapterLaunchError("remote service preflight is absent", process_absent=True)
        try:
            started = _remote_action(
                "start", str(context.scratch_path), request_sha256, timeout=360
            )
            if started.get("process_absent") is True:
                raise AdapterLaunchError(
                    "remote Docker create failed with exact absence", process_absent=True
                )
            if (
                started.get("display_name") != service.DISPLAY_NAME
                or started.get("artifact_name") != service.ARTIFACT_NAME
                or started.get("served_alias") != service.SERVED_ALIAS
            ):
                raise RemoteFlashNextServiceError(
                    "remote start reported the wrong model/artifact identity"
                )
            container_id = str(started.get("container_id") or "")
            pid = started.get("pid")
            ticks = started.get("start_ticks")
            if (
                _CONTAINER.fullmatch(container_id) is None
                or type(pid) is not int
                or pid <= 1
                or type(ticks) is not int
            ):
                raise RemoteFlashNextServiceError("remote start receipt is malformed")
            deadline = time.monotonic() + context.profile.startup_timeout_seconds
            last_heartbeat = 0.0
            while time.monotonic() < deadline:
                context.startup_check()
                now = time.monotonic()
                if now - last_heartbeat >= 30:
                    context.heartbeat(pid, "Qualified remote Flash-Next SGLang is loading")
                    last_heartbeat = now
                status = _remote_action(
                    "status", str(context.scratch_path), request_sha256, timeout=60
                )
                if status.get("process_absent") is True:
                    raise AdapterLaunchError(
                        "remote service became exactly absent", process_absent=True
                    )
                if (
                    status.get("display_name") != service.DISPLAY_NAME
                    or status.get("artifact_name") != service.ARTIFACT_NAME
                    or status.get("served_alias") != service.SERVED_ALIAS
                    or status.get("container_id") != container_id
                    or status.get("pid") != pid
                    or status.get("start_ticks") != ticks
                ):
                    raise RemoteFlashNextServiceError("remote live identity changed during start")
                if status.get("state") == "ready":
                    tunnel_pid, tunnel_ticks = _start_tunnel(
                        context.run_dir,
                        context.runtime_id,
                        request_sha256,
                        context.lease.physical_gpu,
                    )
                    local_port, _remote_port = _ports(context.lease.physical_gpu)
                    identity = (
                        f"{PROCESS_PREFIX}:{context.runtime_id}:{request_sha256}:"
                        f"{container_id}:{service.load_production_binding(verify_release_hashes=False).binding_sha256}:"
                        f"{pid}:{ticks}:{tunnel_pid}:{tunnel_ticks}"
                    )
                    return LaunchResult(
                        pid=pid,
                        process_identity=identity,
                        endpoint=f"http://127.0.0.1:{local_port}/v1",
                    )
                time.sleep(5)
            raise RemoteFlashNextServiceError("remote service startup exceeded its bound")
        except AdapterLaunchError:
            raise
        except RemoteFlashNextTransportError:
            raise
        except BaseException as exc:
            try:
                status = _remote_action(
                    "status", str(context.scratch_path), request_sha256, timeout=60
                )
            except Exception:
                raise
            if status.get("process_absent") is True:
                raise AdapterLaunchError(
                    f"remote service launch failed with exact absence: {exc}",
                    process_absent=True,
                ) from exc
            raise

    @staticmethod
    def _runtime_parts(
        runtime: Mapping[str, Any]
    ) -> tuple[str, str, str, str, int, int, int, int]:
        match = _PROCESS.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None
            or match.group(1) != runtime.get("runtime_id")
            or runtime.get("profile_id") != PROFILE_ID
            or runtime.get("host") != HOST
            or int(match.group(5)) != runtime.get("pid")
            or PurePosixPath(str(runtime.get("run_dir") or "")).parent
            != PurePosixPath(str(RUN_ROOT))
        ):
            raise RemoteFlashNextServiceError("saved remote service identity changed")
        return (
            match.group(1),
            match.group(2),
            match.group(3),
            match.group(4),
            int(match.group(5)),
            int(match.group(6)),
            int(match.group(7)),
            int(match.group(8)),
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        if runtime.get("process_identity") is None:
            try:
                if _recover_pidless_runtime(runtime):
                    return ProbeResult(
                        ProbeState.ABSENT,
                        False,
                        True,
                        "exact remote uncommitted startup was safely recovered",
                        prelaunch_cleanup_verified=True,
                    )
            except RemoteFlashNextTransportError:
                raise
            except (
                RemoteFlashNextServiceError,
                OSError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        try:
            (
                runtime_id,
                request_sha256,
                container_id,
                binding_sha,
                pid,
                ticks,
                tunnel_pid,
                tunnel_ticks,
            ) = self._runtime_parts(runtime)
            binding = service.load_production_binding(verify_release_hashes=False)
            if binding.binding_sha256 != binding_sha:
                raise RemoteFlashNextServiceError("production binding changed")
            status = _remote_action(
                "status", str(runtime["run_dir"]), request_sha256, timeout=60
            )
            if status.get("process_absent") is True:
                return ProbeResult(
                    ProbeState.ABSENT, False, True, "exact remote container is absent"
                )
            if (
                status.get("display_name") != service.DISPLAY_NAME
                or status.get("artifact_name") != service.ARTIFACT_NAME
                or status.get("served_alias") != service.SERVED_ALIAS
                or status.get("container_id") != container_id
                or status.get("pid") != pid
                or status.get("start_ticks") != ticks
            ):
                raise RemoteFlashNextServiceError("remote process receipt changed")
            receipt = _private_json(Path(str(runtime["run_dir"])) / TUNNEL_RECEIPT)
            local_port, remote_port = _ports(runtime.get("physical_gpu"))
            if (
                receipt.get("runtime_id") != runtime_id
                or receipt.get("request_sha256") != request_sha256
                or receipt.get("state") != "active"
                or receipt.get("pid") != tunnel_pid
                or receipt.get("start_ticks") != tunnel_ticks
                or receipt.get("local_port") != local_port
                or receipt.get("remote_port") != remote_port
                or not _tunnel_exact(receipt)
                or not _endpoint_ready(local_port, semantic=True)
            ):
                raise RemoteFlashNextServiceError("remote loopback semantic identity changed")
            state = ProbeState.READY if status.get("state") == "ready" else ProbeState.STARTING
            return ProbeResult(
                state,
                True,
                False,
                f"{service.DISPLAY_NAME} remote primary is ready "
                f"(compatibility wire alias {service.SERVED_ALIAS})"
                if state is ProbeState.READY
                else "qualified remote service is still starting",
            )
        except RemoteFlashNextTransportError:
            raise
        except (RemoteFlashNextServiceError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            runtime_id, request_sha256, _container, _binding, _pid, _ticks, _tpid, _tticks = self._runtime_parts(runtime)
            remote = _remote_action(
                "stop", str(runtime["run_dir"]), request_sha256, timeout=120
            )
            tunnel_absent = _stop_tunnel(
                Path(str(runtime["run_dir"])), runtime_id, request_sha256
            )
            absent = remote.get("process_absent") is True and tunnel_absent
            return StopResult(
                absent,
                True,
                reason if absent else "exact remote container/tunnel is still stopping",
            )
        except RemoteFlashNextTransportError:
            raise
        except (RemoteFlashNextServiceError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return StopResult(False, False, str(exc))

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        scratch = storage.get("scratch_path")
        if scratch != runtime.get("run_dir") or not isinstance(scratch, str):
            raise RemoteFlashNextServiceError("remote service storage identity changed")
        if runtime.get("process_identity") is None:
            if (
                runtime.get("state") not in {"stopped", "lost"}
                or runtime.get("process_absent") not in {True, 1}
                or runtime.get("pid") is not None
                or runtime.get("endpoint") is not None
            ):
                return StorageFinalizationResult(
                    True,
                    False,
                    0,
                    "PID-less remote cleanup lacks Fleet's terminal absence barrier",
                )
            try:
                recovered = _recover_pidless_runtime(
                    runtime, allowed_states=frozenset({"stopped", "lost"})
                )
            except FileNotFoundError:
                runtime_id = str(runtime.get("runtime_id") or "")
                run_dir = Path(scratch)
                tunnel = run_dir / TUNNEL_RECEIPT
                if (
                    _RUNTIME.fullmatch(runtime_id) is None
                    or runtime.get("profile_id") != PROFILE_ID
                    or runtime.get("adapter") != ADAPTER_NAME
                    or runtime.get("host") != HOST
                    or run_dir != Path(RUN_ROOT) / runtime_id
                    or tunnel.exists()
                    or tunnel.is_symlink()
                    or not _remote_run_absent(scratch)
                    or not _remote_named_container_absent(
                        f"aeon-qwen38-flash-next-179-{runtime_id}"
                    )
                ):
                    return StorageFinalizationResult(
                        True,
                        False,
                        0,
                        "PID-less remote preparation cleanup is ambiguous",
                    )
                recovered = True
            if not recovered:
                return StorageFinalizationResult(
                    True, False, 0, "PID-less remote cleanup remains incomplete"
                )
            return StorageFinalizationResult(
                True,
                True,
                0,
                "exact PID-less remote scratch/container absence is proven",
            )
        runtime_id, request_sha256, _container, _binding, _pid, _ticks, _tpid, _tticks = self._runtime_parts(runtime)
        tunnel_path = Path(scratch) / TUNNEL_RECEIPT
        try:
            tunnel = _private_json(tunnel_path)
        except FileNotFoundError:
            return StorageFinalizationResult(
                True, False, 0, "local tunnel receipt is absent; cleanup authority is incomplete"
            )
        if (
            tunnel.get("runtime_id") != runtime_id
            or tunnel.get("request_sha256") != request_sha256
            or tunnel.get("state") != "stopped"
            or not _pid_absent(tunnel.get("pid"))
        ):
            return StorageFinalizationResult(
                True, False, 0, "exact local tunnel absence is not proven"
            )
        container_name = f"aeon-qwen38-flash-next-179-{runtime_id}"
        if _remote_run_absent(scratch):
            if not _remote_named_container_absent(container_name):
                return StorageFinalizationResult(
                    True, False, 0, "remote scratch is absent but container absence is unproven"
                )
            return StorageFinalizationResult(
                True,
                True,
                0,
                "exact remote scratch is already absent; cache references remain Fleet-owned",
            )
        result = _remote_action("cleanup", scratch, request_sha256, timeout=180)
        if (
            result.get("state") != "cleaned"
            or result.get("process_absent") is not True
            or result.get("cache_entries_removed") != 0
            or result.get("docker_images_removed") != 0
            or type(result.get("reclaimed_bytes")) is not int
            or result["reclaimed_bytes"] < 0
        ):
            return StorageFinalizationResult(
                True, False, 0, "remote scratch cleanup proof changed"
            )
        return StorageFinalizationResult(
            True,
            True,
            int(result["reclaimed_bytes"]),
            "exact run scratch removed; Fleet retains/releases cache references separately",
        )


def create_fleet_adapter() -> AeonQwenFlashNextRemoteServiceAdapter:
    return AeonQwenFlashNextRemoteServiceAdapter()
