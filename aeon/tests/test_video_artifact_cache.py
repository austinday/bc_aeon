"""Hermetic promotion safety tests for the video artifact cache backend."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fleet_compute.artifact_cache import ArtifactCacheSafetyError
from fleet_compute.models import ArtifactDescriptor, ArtifactKind

from aeon.core import video_artifact_cache as cache


def _descriptor(payload: bytes) -> ArtifactDescriptor:
    digest = hashlib.sha256(payload).hexdigest()
    return ArtifactDescriptor(
        artifact_id="video-promotion-test",
        identity_key="promotion_test",
        kind=ArtifactKind.FILE,
        canonical_path=(
            "/home/aday/.local/state/fleet-compute/artifacts/"
            "video-promotion-test/payload.bin"
        ),
        digest_sha256=digest,
        size_bytes_max=4096,
        inode_count_max=1,
        transfer_bytes_max=4096,
        cold_peak_bytes_max=4096,
    )


def _release_spec(descriptor: ArtifactDescriptor) -> SimpleNamespace:
    return SimpleNamespace(
        identity_key=descriptor.identity_key,
        kind=descriptor.kind,
        canonical_path=Path(descriptor.canonical_path),
        digest_sha256=descriptor.digest_sha256,
    )


def _execute_embedded(root: Path):
    def execute(
        _host: str, script: str, *arguments: str, **_kwargs: object
    ) -> dict[str, object]:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                os.uname().nodename,
                str(root),
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise ArtifactCacheSafetyError("embedded video worker proof refused")
        value = json.loads(result.stdout)
        assert isinstance(value, dict)
        return value

    return execute


def _temporary_entry(
    root: Path, descriptor: ArtifactDescriptor, payload: bytes
) -> tuple[Path, Path]:
    staging = root / ".staging"
    staging.mkdir(parents=True)
    root.chmod(0o700)
    staging.chmod(0o700)
    temporary = staging / f"{descriptor.digest_sha256}.deadbeef.partial"
    final = (
        root
        / "sha256"
        / descriptor.digest_sha256[:2]
        / descriptor.digest_sha256
    )
    temporary.write_bytes(payload)
    temporary.chmod(0o600)
    os.setxattr(
        temporary,
        cache.OWNERSHIP_XATTR,
        cache._marker(descriptor.kind, descriptor.digest_sha256).encode("utf-8"),
    )
    return temporary, final


def _inspect(
    backend: cache.VideoArtifactCacheBackend,
    root: Path,
    temporary: Path,
    descriptor: ArtifactDescriptor,
    *,
    verify_content: bool,
):
    inspection = backend.inspect_entry(
        host="192.168.0.178",
        path=str(temporary),
        descriptor=descriptor,
        expected_filesystem_id=str(root.stat().st_dev),
        verify_content=verify_content,
    )
    assert inspection is not None and inspection.identity_token is not None
    return inspection


def test_promote_publishes_only_the_exact_descriptor_payload(tmp_path: Path) -> None:
    payload = b"exact reviewed video payload"
    descriptor = _descriptor(payload)
    root = tmp_path / "cache"
    backend = cache.VideoArtifactCacheBackend()
    backend._remote_python = _execute_embedded(root)  # type: ignore[method-assign]

    with (
        patch.object(cache, "VIDEO_WORKER_CACHE_ROOT", root),
        patch.object(
            cache,
            "VIDEO_ARTIFACTS_BY_ID",
            {descriptor.artifact_id: _release_spec(descriptor)},
        ),
    ):
        temporary, final = _temporary_entry(root, descriptor, payload)
        inspection = _inspect(
            backend, root, temporary, descriptor, verify_content=True
        )
        backend.promote(
            host="192.168.0.178",
            temporary_path=str(temporary),
            final_path=str(final),
            descriptor=descriptor,
            identity_token=inspection.identity_token,
            expected_filesystem_id=str(root.stat().st_dev),
            owner_uid=os.geteuid(),
        )

    assert final.read_bytes() == payload
    assert not temporary.exists()


def test_promote_refuses_wrong_content_even_with_current_token(tmp_path: Path) -> None:
    descriptor = _descriptor(b"expected reviewed video payload")
    root = tmp_path / "cache"
    backend = cache.VideoArtifactCacheBackend()
    backend._remote_python = _execute_embedded(root)  # type: ignore[method-assign]

    with (
        patch.object(cache, "VIDEO_WORKER_CACHE_ROOT", root),
        patch.object(
            cache,
            "VIDEO_ARTIFACTS_BY_ID",
            {descriptor.artifact_id: _release_spec(descriptor)},
        ),
    ):
        temporary, final = _temporary_entry(root, descriptor, b"wrong payload")
        inspection = _inspect(
            backend, root, temporary, descriptor, verify_content=False
        )
        with pytest.raises(ArtifactCacheSafetyError):
            backend.promote(
                host="192.168.0.178",
                temporary_path=str(temporary),
                final_path=str(final),
                descriptor=descriptor,
                identity_token=inspection.identity_token,
                expected_filesystem_id=str(root.stat().st_dev),
                owner_uid=os.geteuid(),
            )

    assert temporary.read_bytes() == b"wrong payload"
    assert not final.exists()


def test_promote_refuses_content_changed_after_verified_inspection(
    tmp_path: Path,
) -> None:
    payload = b"exact reviewed video payload"
    descriptor = _descriptor(payload)
    root = tmp_path / "cache"
    backend = cache.VideoArtifactCacheBackend()
    backend._remote_python = _execute_embedded(root)  # type: ignore[method-assign]

    with (
        patch.object(cache, "VIDEO_WORKER_CACHE_ROOT", root),
        patch.object(
            cache,
            "VIDEO_ARTIFACTS_BY_ID",
            {descriptor.artifact_id: _release_spec(descriptor)},
        ),
    ):
        temporary, final = _temporary_entry(root, descriptor, payload)
        inspection = _inspect(
            backend, root, temporary, descriptor, verify_content=True
        )
        temporary.write_bytes(b"mutated after verification")
        with pytest.raises(ArtifactCacheSafetyError):
            backend.promote(
                host="192.168.0.178",
                temporary_path=str(temporary),
                final_path=str(final),
                descriptor=descriptor,
                identity_token=inspection.identity_token,
                expected_filesystem_id=str(root.stat().st_dev),
                owner_uid=os.geteuid(),
            )

    assert temporary.exists()
    assert not final.exists()
