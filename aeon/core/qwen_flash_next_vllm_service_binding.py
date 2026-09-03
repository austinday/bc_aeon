"""Fail-closed promotion binding for the disabled vLLM service lane.

This module is launch-free.  It closes one production candidate over the exact
canary artifacts and qualification receipt; enabling or reloading Fleet remains
a separate explicit operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.scripts import qwen_flash_next_vllm_canary_worker as canary_worker


PROFILE_ID = "aeon-qwen38-flash-next-vllm-177"
SERVICE_ID = "aeon-qwen38-standard"
BINDING_SCHEMA = "aeon-qwen38-flash-next-vllm-service-binding-v1"
BINDING_PATH = Path(
    "/home/aday/.local/state/aeon-flash-next/releases/"
    "vllm-production-service-binding.json"
)
CANARY_OUTPUT_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/artifacts/"
    "aeon-qwen38-flash-next-vllm-canary"
)
CHECKPOINT_ROOT = Path("/home/aday/.local/state/aeon-flash-next/models")
IMAGE_ARCHIVE_ROOT = Path("/home/aday/.local/state/aeon-flash-next/runtime-images")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_FIELDS = {
    "schema_version",
    "complete",
    "profile_id",
    "service_id",
    "host",
    "physical_gpu",
    "vram_cap_gib",
    "runtime",
    "qualification_receipt",
    "qualification_receipt_sha256",
    "checkpoint_path",
    "checkpoint_manifest_path",
    "checkpoint_manifest_sha256",
    "derived_image_digest",
    "derived_image_config_digest",
    "derived_image_archive_path",
    "derived_image_archive_sha256",
    "canary_artifact_identity",
}
CANARY_IDENTITY_FIELDS = {
    "adapter_source",
    "worker_source",
    "harness_source",
    "cuda_sampler_source",
    "runtime_contract_source",
    "source_manifest",
    "checkpoint_manifest",
    "derived_image",
    "derived_image_config",
    "derived_image_archive",
}


class VllmServiceBindingError(RuntimeError):
    """The promotion binding is unsafe, incomplete, or not qualified."""


@dataclass(frozen=True)
class VllmServiceBinding:
    path: Path
    sha256: str
    qualification_receipt: Path
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    derived_image_archive_path: Path
    raw: Mapping[str, Any]

    @property
    def artifact_identity(self) -> dict[str, str]:
        canary_identity = self.raw["canary_artifact_identity"]
        return {
            "binding": self.sha256,
            "qualification": str(self.raw["qualification_receipt_sha256"]),
            **{str(key): str(value) for key, value in canary_identity.items()},
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_file(path: Path, *, maximum: int) -> Mapping[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise VllmServiceBindingError(f"private binding evidence is unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VllmServiceBindingError(f"binding evidence is malformed: {path}") from exc
    if not isinstance(value, Mapping):
        raise VllmServiceBindingError("binding evidence is not an object")
    return value


def _beneath(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise VllmServiceBindingError(f"{label} is outside its canonical root") from exc
    return resolved


def load_binding(path: Path = BINDING_PATH) -> VllmServiceBinding:
    """Load and fully revalidate one exact promotion binding."""

    raw = _private_file(path, maximum=2 * 1024 * 1024)
    if (
        set(raw) != EXPECTED_FIELDS
        or raw.get("schema_version") != BINDING_SCHEMA
        or raw.get("complete") is not True
        or raw.get("profile_id") != PROFILE_ID
        or raw.get("service_id") != SERVICE_ID
        or raw.get("host") != contract.HOST
        or raw.get("physical_gpu") != contract.PHYSICAL_GPU
        or raw.get("vram_cap_gib") != contract.VRAM_CAP_GIB
        or raw.get("runtime") != contract.expected_runtime()
    ):
        raise VllmServiceBindingError("production binding contract changed")
    digests = {
        name: raw.get(name)
        for name in (
            "qualification_receipt_sha256",
            "checkpoint_manifest_sha256",
            "derived_image_config_digest",
            "derived_image_archive_sha256",
        )
    }
    image = str(raw.get("derived_image_digest") or "")
    canary_identity = raw.get("canary_artifact_identity")
    if (
        any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in digests.values())
        or not image.startswith("sha256:")
        or SHA256_RE.fullmatch(image.removeprefix("sha256:")) is None
        or not isinstance(canary_identity, Mapping)
        or set(canary_identity) != CANARY_IDENTITY_FIELDS
        or any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in canary_identity.values())
    ):
        raise VllmServiceBindingError("production artifact identities are incomplete")
    qualification = _beneath(Path(str(raw["qualification_receipt"])), CANARY_OUTPUT_ROOT, "qualification receipt")
    checkpoint = _beneath(Path(str(raw["checkpoint_path"])), CHECKPOINT_ROOT, "checkpoint")
    manifest = _beneath(Path(str(raw["checkpoint_manifest_path"])), checkpoint, "checkpoint manifest")
    archive = _beneath(Path(str(raw["derived_image_archive_path"])), IMAGE_ARCHIVE_ROOT, "image archive")
    if (
        _sha256(qualification) != raw["qualification_receipt_sha256"]
        or _sha256(manifest) != raw["checkpoint_manifest_sha256"]
        or _sha256(archive) != raw["derived_image_archive_sha256"]
        or canary_identity["checkpoint_manifest"] != raw["checkpoint_manifest_sha256"]
        or canary_identity["derived_image"] != image.removeprefix("sha256:")
        or canary_identity["derived_image_config"] != raw["derived_image_config_digest"]
        or canary_identity["derived_image_archive"] != raw["derived_image_archive_sha256"]
    ):
        raise VllmServiceBindingError("bound canary artifact bytes changed")
    if manifest != checkpoint / "SHA256SUMS":
        raise VllmServiceBindingError("checkpoint binding is not the full SHA256SUMS closure")
    try:
        canary_worker._verify_checkpoint_manifest(
            {
                "checkpoint_path": str(checkpoint),
                "checkpoint_manifest_path": str(manifest),
                "checkpoint_manifest_sha256": raw["checkpoint_manifest_sha256"],
            }
        )
        archive_manifest, archive_config = canary_worker._oci_identity(archive)
    except canary_worker.CanaryWorkerError as exc:
        raise VllmServiceBindingError(str(exc)) from exc
    if (
        archive_manifest != raw["derived_image_digest"]
        or archive_config != raw["derived_image_config_digest"]
    ):
        raise VllmServiceBindingError("derived OCI manifest/config identity changed")
    # The production container mounts these reviewed canary/runtime sources from
    # the live Aeon tree.  Recompute their closure instead of trusting the hashes
    # merely because they were copied into the binding.
    from aeon.core import qwen_flash_next_vllm_canary_adapter as canary_adapter

    current_canary_identity = canary_adapter.expected_artifact_identity(
        {
            "checkpoint_manifest_sha256": str(
                raw["checkpoint_manifest_sha256"]
            ),
            "derived_image_digest": str(raw["derived_image_digest"]),
            "derived_image_config_digest": str(
                raw["derived_image_config_digest"]
            ),
            "derived_image_archive_sha256": str(
                raw["derived_image_archive_sha256"]
            ),
        }
    )
    if current_canary_identity != canary_identity:
        raise VllmServiceBindingError("bound canary runtime sources changed")
    receipt = _private_file(qualification, maximum=64 * 1024 * 1024)
    failures = contract.validate_qualification_receipt(receipt)
    if failures:
        raise VllmServiceBindingError(
            "qualification receipt is not promotable: " + "; ".join(failures)
        )
    return VllmServiceBinding(
        path=path,
        sha256=_sha256(path),
        qualification_receipt=qualification,
        checkpoint_path=checkpoint,
        checkpoint_manifest_path=manifest,
        derived_image_archive_path=archive,
        raw=dict(raw),
    )
