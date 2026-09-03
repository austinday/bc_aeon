"""Hermetic safety tests for Aeon's Fleet-owned Qwen artifact cache."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fleet_compute.artifact_cache import ArtifactCacheError, ArtifactCacheSafetyError
from fleet_compute.models import ArtifactDescriptor, ArtifactKind

from aeon.core import qwen_artifact_cache as cache
from aeon.core import qwen_fleet_runtime as fleet
from aeon.core.qwen_capabilities import qwen_runtime_capability
from aeon.core.qwen_runtime import QwenRuntimeError
from aeon.scripts import qwen_remote_worker as worker


def _oci_descriptor(
    digest: str = fleet.QWEN_STANDARD_IMAGE_CONFIG_SHA256,
) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=fleet.QWEN_IMAGE_CACHE_ARTIFACT_ID,
        identity_key="image",
        kind=ArtifactKind.OCI_ARCHIVE,
        canonical_path=str(cache.CANONICAL_OCI_ROOT / f"{digest}.tar"),
        digest_sha256=digest,
        size_bytes_max=65_536,
        inode_count_max=1,
        transfer_bytes_max=fleet.QWEN_IMAGE_ARCHIVE_MAX_BYTES,
        cold_peak_bytes_max=26_318_824_199,
    )


def _model_descriptor() -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=fleet.QWEN_MODEL_CACHE_ARTIFACT_ID,
        identity_key="model_sha256s",
        kind=ArtifactKind.MANIFESTED_TREE,
        canonical_path=str(cache.CANONICAL_MODEL_ROOT),
        digest_sha256=cache.CANONICAL_MODEL_SHA256SUMS,
        size_bytes_max=20_600_000_000,
        inode_count_max=100_000,
        transfer_bytes_max=20_600_000_000,
        cold_peak_bytes_max=20_600_000_000,
        manifest_path=str(cache.CANONICAL_MODEL_ROOT / "SHA256SUMS"),
        manifest_format="sha256sum-v1",
    )


def _small_tree_descriptor() -> tuple[ArtifactDescriptor, bytes, bytes]:
    payload = b"small exact cache payload\n"
    manifest = (
        hashlib.sha256(payload).hexdigest().encode("ascii")
        + b"  nested/payload.bin\n"
    )
    digest = hashlib.sha256(manifest).hexdigest()
    canonical = Path("/home/aday/.local/state/fleet-compute/test-qwen-tree")
    return (
        ArtifactDescriptor(
            artifact_id=fleet.QWEN_SOURCE_CACHE_ARTIFACT_ID,
            identity_key="runtime_source",
            kind=ArtifactKind.MANIFESTED_TREE,
            canonical_path=str(canonical),
            digest_sha256=digest,
            size_bytes_max=1_000_000,
            inode_count_max=100,
            transfer_bytes_max=1_000_000,
            cold_peak_bytes_max=1_000_000,
            manifest_path=str(canonical / "MANIFEST"),
            manifest_format="sha256sum-v1",
        ),
        manifest,
        payload,
    )


def _write_small_owned_tree(
    root: Path, path: Path, descriptor: ArtifactDescriptor
) -> None:
    _descriptor, manifest, payload = _small_tree_descriptor()
    path.mkdir(parents=True)
    (path / "nested").mkdir()
    (path / "MANIFEST").write_bytes(manifest)
    (path / "nested" / "payload.bin").write_bytes(payload)
    for directory in (path, path / "nested"):
        directory.chmod(0o700)
    for item in (path / "MANIFEST", path / "nested" / "payload.bin"):
        item.chmod(0o600)
    marker = cache._ownership_value(
        ArtifactKind.MANIFESTED_TREE, descriptor.digest_sha256
    ).replace(str(fleet.FLEET_WORKER_CACHE_ROOT), str(root))
    os.setxattr(path, cache.OWNERSHIP_XATTR, marker.encode())


def _cache_request() -> dict[str, object]:
    source_digest = "1" * 64
    model_digest = "2" * 64
    image_digest = "3" * 64

    def binding(
        artifact_id: str,
        kind: str,
        digest: str,
        size: int,
        inodes: int,
        payload: str | None = None,
    ) -> dict[str, object]:
        value: dict[str, object] = {
            "artifact_id": artifact_id,
            "kind": kind,
            "worker_path": str(
                fleet.FLEET_WORKER_CACHE_ROOT / "sha256" / digest[:2] / digest
            ),
            "digest_sha256": digest,
            "size_bytes": size,
            "inode_count": inodes,
            "filesystem_id": "123",
        }
        if payload is not None:
            value["payload_sha256"] = payload
        return value

    return {
        "schema_version": 1,
        "source": binding(
            fleet.QWEN_SOURCE_CACHE_ARTIFACT_ID,
            "manifested_tree",
            source_digest,
            100,
            2,
        ),
        "model": binding(
            fleet.QWEN_MODEL_CACHE_ARTIFACT_ID,
            "manifested_tree",
            model_digest,
            200,
            2,
        ),
        "image": binding(
            fleet.QWEN_IMAGE_CACHE_ARTIFACT_ID,
            "oci_archive",
            image_digest,
            200,
            1,
            payload="4" * 64,
        ),
    }


def _tar_member(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o600
    bundle.addfile(member, io.BytesIO(payload))


def _docker_archive(path: Path, *, extra: bool = False, second_image: bool = False) -> str:
    config_payload = json.dumps(
        {"config": {"Env": [], "ExposedPorts": {"8000/tcp": {}}}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(config_payload).hexdigest()
    config_name = f"{digest}.json"
    item = {"Config": config_name, "RepoTags": None, "Layers": ["layer/layer.tar"]}
    manifest = [item, dict(item)] if second_image else [item]
    with tarfile.open(path, "w") as bundle:
        directory = tarfile.TarInfo("layer/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o700
        bundle.addfile(directory)
        _tar_member(bundle, "manifest.json", json.dumps(manifest).encode("utf-8"))
        _tar_member(bundle, config_name, config_payload)
        _tar_member(bundle, "layer/layer.tar", b"layer")
        if extra:
            _tar_member(bundle, "unreferenced.bin", b"unknown")
    path.chmod(0o600)
    return digest


def _execute_embedded(
    root: Path,
):
    def execute(
        _host: str, script: str, *arguments: str, **_kwargs: object
    ) -> dict[str, object]:
        result = subprocess.run(
            [
                os.environ.get("PYTHON", "/usr/bin/python3"),
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
            raise ArtifactCacheSafetyError("embedded worker proof refused")
        value = json.loads(result.stdout)
        assert isinstance(value, dict)
        return value

    return execute


def test_unqualified_worker_is_refused() -> None:
    with pytest.raises(ArtifactCacheSafetyError):
        cache.AeonQwenArtifactBackend._host("192.168.0.179")


def test_docker_archive_requires_one_exact_member_closure(tmp_path: Path) -> None:
    valid = tmp_path / "valid.tar"
    digest = _docker_archive(valid)
    assert cache._validate_oci_archive(valid, digest=digest, maximum_bytes=10_000_000)

    for label, kwargs in (
        ("extra", {"extra": True}),
        ("second", {"second_image": True}),
    ):
        candidate = tmp_path / f"{label}.tar"
        candidate_digest = _docker_archive(candidate, **kwargs)
        with pytest.raises(ArtifactCacheSafetyError):
            cache._validate_oci_archive(
                candidate, digest=candidate_digest, maximum_bytes=10_000_000
            )


def test_canonical_checksum_uses_one_inherited_anchored_fd(tmp_path: Path) -> None:
    payload = b"anchored canonical archive"
    path = tmp_path / "archive.tar"
    path.write_bytes(payload)
    path.chmod(0o600)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        digest = cache.AeonQwenArtifactBackend()._fd_sha256(
            descriptor,
            progress=lambda *_args: None,
            total=1024,
            expected_link_count=1,
        )
    finally:
        os.close(descriptor)
    assert digest == hashlib.sha256(payload).hexdigest()


def test_warm_exact_image_skips_archive_transfer_and_load() -> None:
    descriptor = _oci_descriptor()
    backend = cache.AeonQwenArtifactBackend()
    progress: list[tuple[int, int]] = []
    image = {"Id": f"sha256:{descriptor.digest_sha256}", "Size": 1234, "Config": {}}
    with (
        patch.object(cache, "local_image_id", return_value=image["Id"]),
        patch.object(cache, "local_image_size", return_value=image["Size"]),
        patch.object(backend, "_prepare_remote_temporary"),
        patch.object(backend, "_remote_image_inspection", return_value=image),
        patch.object(backend, "_commit_remote_oci_receipt") as commit,
        patch.object(backend, "_canonical_archive") as canonical,
        patch.object(backend, "_run_with_progress") as run,
    ):
        backend.stage(
            host="192.168.0.180",
            descriptor=descriptor,
            temporary_path=str(
                fleet.FLEET_WORKER_CACHE_ROOT
                / ".staging"
                / f"{descriptor.digest_sha256}.nonce.partial"
            ),
            expected_filesystem_id="123",
            max_bytes_per_second=10_000_000,
            progress=lambda done, total: progress.append((done, total)),
        )
    canonical.assert_not_called()
    run.assert_not_called()
    commit.assert_called_once()
    assert commit.call_args.args[-1]["archive_payload_sha256"] == descriptor.digest_sha256
    assert progress == [(descriptor.transfer_bytes_max, descriptor.transfer_bytes_max)]


def test_remote_image_absence_must_be_exact_not_transport_ambiguity() -> None:
    image_id = "sha256:" + "a" * 64
    missing = subprocess.CompletedProcess(
        [],
        1,
        stdout="[]\n",
        stderr=f"Error response from daemon: No such image: {image_id}\n",
    )
    assert (
        cache.AeonQwenArtifactBackend(command_runner=lambda *a, **k: missing)
        ._remote_image_inspection("192.168.0.180", image_id)
        is None
    )
    ambiguous = subprocess.CompletedProcess([], 255, stdout="", stderr="connection lost")
    with pytest.raises(ArtifactCacheSafetyError):
        cache.AeonQwenArtifactBackend(
            command_runner=lambda *a, **k: ambiguous
        )._remote_image_inspection("192.168.0.180", image_id)


def test_runtime_binding_rejects_oci_receipt_above_64k() -> None:
    request = _cache_request()
    request["image"]["size_bytes"] = 65_537  # type: ignore[index]
    with pytest.raises(QwenRuntimeError):
        fleet._validated_artifact_cache_request(request)


def test_descriptor_bounds_and_identity_are_exact() -> None:
    backend = cache.AeonQwenArtifactBackend()
    model = _model_descriptor()
    image = _oci_descriptor()
    backend._validate_descriptor(model)
    backend._validate_descriptor(image)
    for changed in (
        replace(model, size_bytes_max=model.size_bytes_max - 1),
        replace(model, identity_key="model"),
        replace(image, size_bytes_max=fleet.QWEN_IMAGE_ARCHIVE_MAX_BYTES),
        replace(image, canonical_path=str(cache.CANONICAL_OCI_ROOT / "image.tar")),
    ):
        with pytest.raises(ArtifactCacheSafetyError):
            backend._validate_descriptor(changed)


def test_model_origin_allows_verified_hardlinks_but_source_origin_does_not() -> None:
    state_root = Path("/home/aday/.local/state/fleet-compute")
    with tempfile.TemporaryDirectory(
        prefix="test-qwen-hardlink-", dir=state_root
    ) as temporary:
        base = Path(temporary)
        root = base / "model"
        root.mkdir(mode=0o700)
        payload = root / "model-shard.bin"
        payload.write_bytes(b"immutable model shard")
        payload.chmod(0o600)
        os.link(payload, base / "shared-model-shard.bin")
        manifest_payload = (
            hashlib.sha256(payload.read_bytes()).hexdigest()
            + "  model-shard.bin\n"
        ).encode()
        manifest = root / "SHA256SUMS"
        manifest.write_bytes(manifest_payload)
        manifest.chmod(0o600)
        descriptor = ArtifactDescriptor(
            artifact_id=fleet.QWEN_MODEL_CACHE_ARTIFACT_ID,
            identity_key="model_sha256s",
            kind=ArtifactKind.MANIFESTED_TREE,
            canonical_path=str(root),
            digest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
            size_bytes_max=1_000_000,
            inode_count_max=10,
            transfer_bytes_max=1_000_000,
            cold_peak_bytes_max=1_000_000,
            manifest_path=str(manifest),
            manifest_format="sha256sum-v1",
        )

        assert cache.AeonQwenArtifactBackend._manifest_files(descriptor) == (
            "model-shard.bin",
            "SHA256SUMS",
        )
        with pytest.raises(ArtifactCacheSafetyError):
            cache.AeonQwenArtifactBackend._manifest_files(
                replace(
                    descriptor,
                    artifact_id=fleet.QWEN_SOURCE_CACHE_ARTIFACT_ID,
                    identity_key="runtime_source",
                )
            )


def test_remote_preflight_startup_refusal_terminates_exact_child() -> None:
    class Input:
        def write(self, _payload: str) -> None:
            return None

        def close(self) -> None:
            return None

    class Process:
        stdin: Input | None = Input()
        returncode: int | None = None
        terminated = False
        reaped = False

        def communicate(self, timeout: float | None = None):
            raise subprocess.TimeoutExpired(["remote"], timeout)

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = -15
            self.reaped = True
            return self.returncode

    capability, _manifest = qwen_runtime_capability(
        "qwen38-compact-180-128k", require_enabled=True
    )
    process = Process()
    checks = 0

    def startup_check() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("startup lease refused")

    with pytest.raises(RuntimeError, match="startup lease refused"):
        fleet.remote_call(
            capability,
            "1" * 64,
            "preflight",
            {"artifact_cache": _cache_request()},
            timeout=1800,
            startup_check=startup_check,
            popen_factory=lambda *args, **kwargs: process,  # type: ignore[arg-type]
        )
    assert process.terminated and process.reaped and process.returncode == -15


def test_progress_refusal_terminates_and_reaps_exact_child() -> None:
    class Process:
        returncode: int | None = None
        terminated = False
        reaped = False
        calls = 0

        def communicate(self, timeout: float | None = None):
            self.calls += 1
            if not self.terminated:
                raise subprocess.TimeoutExpired(["transfer"], timeout)
            self.returncode = -15
            self.reaped = True
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.terminated = True

    process = Process()
    callbacks = 0

    def refuse(_done: int, _total: int) -> None:
        nonlocal callbacks
        callbacks += 1
        if callbacks == 2:
            raise RuntimeError("heartbeat refused")

    backend = cache.AeonQwenArtifactBackend(
        popen_factory=lambda *a, **k: process, clock=lambda: 0
    )
    with pytest.raises(RuntimeError, match="heartbeat refused"):
        backend._run_with_progress(
            ["transfer"], progress=refuse, total=10, progress_probe=lambda: 1
        )
    assert process.terminated and process.reaped and process.returncode == -15


def test_transfer_timeout_never_uses_unbounded_child_wait() -> None:
    class Process:
        returncode: int | None = None
        terminated = False
        killed = False
        reaped = False
        waits: list[float | None] = []

        def communicate(self, timeout: float | None = None):
            self.waits.append(timeout)
            if not self.killed:
                raise subprocess.TimeoutExpired(["transfer"], timeout)
            self.returncode = -9
            self.reaped = True
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    process = Process()
    times = iter((0.0, 2.0))
    backend = cache.AeonQwenArtifactBackend(
        popen_factory=lambda *args, **kwargs: process,
        clock=lambda: next(times),
    )
    with pytest.raises(ArtifactCacheError, match="timed out"):
        backend._run_with_progress(
            ["transfer"], progress=lambda *_args: None, total=10, timeout=1
        )
    assert process.terminated and process.killed and process.reaped
    assert process.waits == [60, 5, 5]
    assert all(wait is not None for wait in process.waits)


def test_promote_is_atomic_no_clobber_and_never_deletes_destination() -> None:
    descriptor = _oci_descriptor()
    scripts: list[str] = []
    backend = cache.AeonQwenArtifactBackend()

    def remote(_host: str, script: str, *_args: str, **_kwargs: object):
        scripts.append(script)
        return {"ok": True}

    backend._remote_python = remote  # type: ignore[method-assign]
    temporary = (
        fleet.FLEET_WORKER_CACHE_ROOT
        / ".staging"
        / f"{descriptor.digest_sha256}.nonce.partial"
    )
    final = (
        fleet.FLEET_WORKER_CACHE_ROOT
        / "sha256"
        / descriptor.digest_sha256[:2]
        / descriptor.digest_sha256
    )
    backend.promote(
        host="192.168.0.180",
        temporary_path=str(temporary),
        final_path=str(final),
        descriptor=descriptor,
        identity_token="b" * 64,
        expected_filesystem_id="123",
        owner_uid=os.geteuid(),
    )
    source = scripts[-1]
    assert "renameat2" in source and "EEXIST" in source
    assert "unlink(" not in source and "rmdir(" not in source


def test_promote_race_leaves_existing_final_and_temporary_untouched(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    staging = root / ".staging"
    descriptor = _oci_descriptor()
    digest = descriptor.digest_sha256
    temporary = staging / f"{digest}.deadbeef.partial"
    final = root / "sha256" / digest[:2] / digest
    final.parent.mkdir(parents=True)
    staging.mkdir()
    receipt = cache._canonical_json(
        {
            "schema_version": 1,
            "image_id": f"sha256:{digest}",
            "image_size_bytes": 1234,
            "archive_payload_sha256": "b" * 64,
        }
    )
    temporary.write_bytes(receipt)
    final.write_bytes(b"pre-existing")
    for directory in (root, staging, root / "sha256", final.parent):
        directory.chmod(0o700)
    temporary.chmod(0o600)
    final.chmod(0o600)
    marker = cache._ownership_value(ArtifactKind.OCI_ARCHIVE, digest).replace(
        str(fleet.FLEET_WORKER_CACHE_ROOT), str(root)
    )
    os.setxattr(temporary, cache.OWNERSHIP_XATTR, marker.encode())
    backend = cache.AeonQwenArtifactBackend()
    backend._remote_python = _execute_embedded(root)  # type: ignore[method-assign]
    with patch.object(cache, "FLEET_WORKER_CACHE_ROOT", root), pytest.raises(
        ArtifactCacheSafetyError
    ):
        with patch.object(
            backend,
            "_remote_image_inspection",
            return_value={"Id": f"sha256:{digest}", "Size": 1234, "Config": {}},
        ):
            inspection = backend.inspect_entry(
                host="192.168.0.180",
                path=str(temporary),
                descriptor=descriptor,
                expected_filesystem_id=str(root.stat().st_dev),
                verify_content=True,
            )
        assert inspection is not None and inspection.identity_token is not None
        backend.promote(
            host="192.168.0.180",
            temporary_path=str(temporary),
            final_path=str(final),
            descriptor=descriptor,
            identity_token=inspection.identity_token,
            expected_filesystem_id=str(root.stat().st_dev),
            owner_uid=os.geteuid(),
        )
    assert final.read_bytes() == b"pre-existing"
    assert temporary.read_bytes() == receipt


def test_promote_refuses_content_changed_after_verified_inspection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    staging = root / ".staging"
    descriptor = _oci_descriptor()
    digest = descriptor.digest_sha256
    temporary = staging / f"{digest}.deadbeef.partial"
    final = root / "sha256" / digest[:2] / digest
    final.parent.mkdir(parents=True)
    staging.mkdir()
    for directory in (root, staging, root / "sha256", final.parent):
        directory.chmod(0o700)
    receipt = {
        "schema_version": 1,
        "image_id": f"sha256:{digest}",
        "image_size_bytes": 1234,
        "archive_payload_sha256": "b" * 64,
    }
    temporary.write_bytes(cache._canonical_json(receipt))
    temporary.chmod(0o600)
    marker = cache._ownership_value(ArtifactKind.OCI_ARCHIVE, digest).replace(
        str(fleet.FLEET_WORKER_CACHE_ROOT), str(root)
    )
    os.setxattr(temporary, cache.OWNERSHIP_XATTR, marker.encode())
    backend = cache.AeonQwenArtifactBackend()
    backend._remote_python = _execute_embedded(root)  # type: ignore[method-assign]
    filesystem_id = str(root.stat().st_dev)
    with patch.object(cache, "FLEET_WORKER_CACHE_ROOT", root), patch.object(
        backend,
        "_remote_image_inspection",
        return_value={"Id": f"sha256:{digest}", "Size": 1234, "Config": {}},
    ):
        inspection = backend.inspect_entry(
            host="192.168.0.180",
            path=str(temporary),
            descriptor=descriptor,
            expected_filesystem_id=filesystem_id,
            verify_content=True,
        )
        assert inspection is not None and inspection.identity_token
        receipt["image_size_bytes"] = 4321
        temporary.write_bytes(cache._canonical_json(receipt))
        with pytest.raises(ArtifactCacheSafetyError):
            backend.promote(
                host="192.168.0.180",
                temporary_path=str(temporary),
                final_path=str(final),
                descriptor=descriptor,
                identity_token=inspection.identity_token,
                expected_filesystem_id=filesystem_id,
                owner_uid=os.geteuid(),
            )
    assert temporary.exists() and not final.exists()


def test_promote_refuses_nested_addition_after_content_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    descriptor, _manifest, _payload = _small_tree_descriptor()
    staging = root / ".staging"
    temporary = staging / f"{descriptor.digest_sha256}.deadbeef.partial"
    final = root / "sha256" / descriptor.digest_sha256[:2] / descriptor.digest_sha256
    _write_small_owned_tree(root, temporary, descriptor)
    root.chmod(0o700)
    staging.chmod(0o700)
    backend = cache.AeonQwenArtifactBackend()
    execute = _execute_embedded(root)
    backend._remote_python = execute  # type: ignore[method-assign]
    filesystem_id = str(root.stat().st_dev)
    with patch.object(cache, "FLEET_WORKER_CACHE_ROOT", root), patch.object(
        backend, "_validate_descriptor"
    ):
        inspection = backend.inspect_entry(
            host="192.168.0.180",
            path=str(temporary),
            descriptor=descriptor,
            expected_filesystem_id=filesystem_id,
            verify_content=True,
        )
        assert inspection is not None and inspection.identity_token

        def mutate_after_hash(host, script, *arguments, **kwargs):
            replacement = (
                '            injected=pathlib.Path(str(temp))/"nested"/"late.bin"\n'
                '            injected.write_bytes(b"late mutation")\n'
                "            injected.chmod(0o600)\n"
                "            # TEST_RACE_BARRIER_PROMOTE_POST_HASH"
            )
            assert "# TEST_RACE_BARRIER_PROMOTE_POST_HASH" in script
            return execute(
                host,
                script.replace(
                    "            # TEST_RACE_BARRIER_PROMOTE_POST_HASH",
                    replacement,
                ),
                *arguments,
                **kwargs,
            )

        backend._remote_python = mutate_after_hash  # type: ignore[method-assign]
        with pytest.raises(ArtifactCacheSafetyError):
            backend.promote(
                host="192.168.0.180",
                temporary_path=str(temporary),
                final_path=str(final),
                descriptor=descriptor,
                identity_token=inspection.identity_token,
                expected_filesystem_id=filesystem_id,
                owner_uid=os.geteuid(),
            )
    assert (temporary / "nested" / "payload.bin").exists()
    assert (temporary / "nested" / "late.bin").exists()
    assert not final.exists()


def test_remove_refuses_nested_addition_after_content_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    descriptor, _manifest, _payload = _small_tree_descriptor()
    final = root / "sha256" / descriptor.digest_sha256[:2] / descriptor.digest_sha256
    _write_small_owned_tree(root, final, descriptor)
    for directory in (root, root / "sha256", final.parent):
        directory.chmod(0o700)
    backend = cache.AeonQwenArtifactBackend()
    execute = _execute_embedded(root)
    backend._remote_python = execute  # type: ignore[method-assign]
    filesystem_id = str(root.stat().st_dev)
    with patch.object(cache, "FLEET_WORKER_CACHE_ROOT", root), patch.object(
        backend, "_validate_descriptor"
    ):
        inspection = backend.inspect_entry(
            host="192.168.0.180",
            path=str(final),
            descriptor=descriptor,
            expected_filesystem_id=filesystem_id,
            verify_content=True,
        )
        assert inspection is not None and inspection.identity_token

        def mutate_after_hash(host, script, *arguments, **kwargs):
            replacement = (
                '            injected=pathlib.Path(str(path))/"nested"/"late.bin"\n'
                '            injected.write_bytes(b"late mutation")\n'
                "            injected.chmod(0o600)\n"
                "            # TEST_RACE_BARRIER_REMOVE_POST_HASH"
            )
            assert "# TEST_RACE_BARRIER_REMOVE_POST_HASH" in script
            return execute(
                host,
                script.replace(
                    "            # TEST_RACE_BARRIER_REMOVE_POST_HASH",
                    replacement,
                ),
                *arguments,
                **kwargs,
            )

        backend._remote_python = mutate_after_hash  # type: ignore[method-assign]
        with pytest.raises(ArtifactCacheSafetyError):
            backend.remove(
                host="192.168.0.180",
                path=str(final),
                descriptor=descriptor,
                identity_token=inspection.identity_token,
                expected_filesystem_id=filesystem_id,
                owner_uid=os.geteuid(),
            )
    assert (final / "MANIFEST").exists()
    assert (final / "nested" / "payload.bin").exists()
    assert (final / "nested" / "late.bin").exists()


def test_remove_exact_nested_tree_revalidates_and_cleans_all_members(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    descriptor, manifest, payload = _small_tree_descriptor()
    final = root / "sha256" / descriptor.digest_sha256[:2] / descriptor.digest_sha256
    _write_small_owned_tree(root, final, descriptor)
    for directory in (root, root / "sha256", final.parent):
        directory.chmod(0o700)
    backend = cache.AeonQwenArtifactBackend()
    backend._remote_python = _execute_embedded(root)  # type: ignore[method-assign]
    filesystem_id = str(root.stat().st_dev)
    with patch.object(cache, "FLEET_WORKER_CACHE_ROOT", root), patch.object(
        backend, "_validate_descriptor"
    ):
        inspection = backend.inspect_entry(
            host="192.168.0.180",
            path=str(final),
            descriptor=descriptor,
            expected_filesystem_id=filesystem_id,
            verify_content=True,
        )
        assert inspection is not None and inspection.identity_token
        removed = backend.remove(
            host="192.168.0.180",
            path=str(final),
            descriptor=descriptor,
            identity_token=inspection.identity_token,
            expected_filesystem_id=filesystem_id,
            owner_uid=os.geteuid(),
        )
    assert removed.removed
    assert removed.reclaimed_bytes == len(manifest) + len(payload)
    assert removed.reclaimed_inodes == 4
    assert not final.exists()


def test_partial_tree_resumes_missing_files_but_token_refuses_later_addition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    staging = root / ".staging"
    digest = cache.CANONICAL_MODEL_SHA256SUMS
    partial = staging / f"{digest}.deadbeef.partial"
    partial.mkdir(parents=True)
    for directory in (root, staging, partial):
        directory.chmod(0o700)
    manifest = partial / "SHA256SUMS"
    manifest.write_bytes((cache.CANONICAL_MODEL_ROOT / "SHA256SUMS").read_bytes())
    manifest.chmod(0o600)
    marker = cache._ownership_value(ArtifactKind.MANIFESTED_TREE, digest).replace(
        str(fleet.FLEET_WORKER_CACHE_ROOT), str(root)
    )
    os.setxattr(partial, cache.OWNERSHIP_XATTR, marker.encode())
    descriptor = _model_descriptor()
    backend = cache.AeonQwenArtifactBackend()
    backend._remote_python = _execute_embedded(root)  # type: ignore[method-assign]
    filesystem_id = str(root.stat().st_dev)
    with patch.object(cache, "FLEET_WORKER_CACHE_ROOT", root):
        assert (
            backend.inspect_entry(
                host="192.168.0.180",
                path=str(partial),
                descriptor=descriptor,
                expected_filesystem_id=filesystem_id,
                verify_content=True,
            )
            is None
        )
        extra = partial / "safe-extra"
        extra.write_bytes(b"first")
        extra.chmod(0o600)
        inspection = backend.inspect_entry(
            host="192.168.0.180",
            path=str(partial),
            descriptor=descriptor,
            expected_filesystem_id=filesystem_id,
            verify_content=False,
        )
        assert inspection is not None and inspection.identity_token
        later = partial / "later-addition"
        later.write_bytes(b"unknown")
        later.chmod(0o600)
        with pytest.raises(ArtifactCacheSafetyError):
            backend.remove(
                host="192.168.0.180",
                path=str(partial),
                descriptor=descriptor,
                identity_token=inspection.identity_token,
                expected_filesystem_id=filesystem_id,
                owner_uid=os.geteuid(),
            )
        assert manifest.exists() and extra.exists() and later.exists()

        current = backend.inspect_entry(
            host="192.168.0.180",
            path=str(partial),
            descriptor=descriptor,
            expected_filesystem_id=filesystem_id,
            verify_content=False,
        )
        assert current is not None and current.identity_token
        backend.remove(
            host="192.168.0.180",
            path=str(partial),
            descriptor=descriptor,
            identity_token=current.identity_token,
            expected_filesystem_id=filesystem_id,
            owner_uid=os.geteuid(),
        )
    assert not partial.exists()


def test_canonical_materialization_is_capacity_gated_and_crash_anonymous(
    tmp_path: Path,
) -> None:
    root = tmp_path / "oci"
    root.mkdir()
    root.chmod(0o700)
    archive_source = tmp_path / "release.tar"
    digest = _docker_archive(archive_source)
    archive_payload = archive_source.read_bytes()
    descriptor = SimpleNamespace(
        digest_sha256=digest,
        canonical_path=str(root / f"{digest}.tar"),
        transfer_bytes_max=fleet.QWEN_IMAGE_ARCHIVE_MAX_BYTES,
    )
    backend = cache.AeonQwenArtifactBackend()
    with (
        patch.object(cache, "CANONICAL_OCI_ROOT", root),
        patch.object(
            cache,
            "_open_canonical_root",
            side_effect=lambda **_kwargs: os.open(
                root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
            ),
        ),
        patch.object(cache, "local_image_id", return_value=f"sha256:{digest}"),
        patch.object(cache, "local_image_size", return_value=1234),
        patch.object(
            cache.os,
            "fstatvfs",
            return_value=SimpleNamespace(f_bavail=1, f_frsize=1, f_favail=100_000),
        ),
        patch.object(backend, "_run_with_progress") as run,
    ):
        with pytest.raises(ArtifactCacheError):
            backend._canonical_archive(descriptor, progress=lambda *_: None)
    run.assert_not_called()
    assert list(root.iterdir()) == []

    def interrupted(_command, **kwargs):
        kwargs["stdout"].write(archive_payload[:1024])
        kwargs["stdout"].flush()
        raise RuntimeError("simulated crash")

    with (
        patch.object(cache, "CANONICAL_OCI_ROOT", root),
        patch.object(
            cache,
            "_open_canonical_root",
            side_effect=lambda **_kwargs: os.open(
                root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
            ),
        ),
        patch.object(cache, "local_image_id", return_value=f"sha256:{digest}"),
        patch.object(cache, "local_image_size", return_value=1234),
        patch.object(backend, "_run_with_progress", side_effect=interrupted),
    ):
        with pytest.raises(RuntimeError, match="simulated crash"):
            backend._canonical_archive(descriptor, progress=lambda *_: None)
    assert list(root.iterdir()) == []

    raw_sha = hashlib.sha256(archive_payload).hexdigest()

    def completed(_command, **kwargs):
        kwargs["stdout"].write(archive_payload)
        kwargs["stdout"].flush()
        return "", ""

    with (
        patch.object(cache, "CANONICAL_OCI_ROOT", root),
        patch.object(
            cache,
            "_open_canonical_root",
            side_effect=lambda **_kwargs: os.open(
                root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
            ),
        ),
        patch.object(cache, "local_image_id", return_value=f"sha256:{digest}"),
        patch.object(cache, "local_image_size", return_value=1234),
        patch.object(backend, "_run_with_progress", side_effect=completed),
        patch.object(backend, "_fd_sha256", return_value=raw_sha),
    ):
        result = backend._canonical_archive(descriptor, progress=lambda *_: None)
    assert result == (root / f"{digest}.tar", raw_sha, 1234)
    assert [item.name for item in root.iterdir()] == [f"{digest}.tar"]


def test_cached_worker_command_uses_isolated_verified_bootstrap() -> None:
    capability, _manifest = qwen_runtime_capability(
        "qwen38-compact-180-128k", require_enabled=True
    )
    request = _cache_request()
    command = fleet._remote_command(
        capability,
        "1" * 64,
        "preflight",
        request,
    )
    assert "-I" in command and "-B" in command
    assert not any(item.startswith("PYTHONPATH=") for item in command)
    assert "runpy.run_path" in command[command.index("-c") + 1]


def test_worker_cache_proof_rejects_symlink_root(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    digest = "a" * 64
    entry = root / "sha256" / digest[:2] / digest
    entry.mkdir(parents=True)
    for directory in (root, root / "sha256", root / "sha256" / digest[:2], entry):
        directory.chmod(0o700)
    marker = {
        "schema_version": 1,
        "kind": "manifested_tree",
        "digest_sha256": digest,
        "cache_root": str(root),
    }
    os.setxattr(
        entry,
        "user.fleet_compute_cache",
        (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    binding = {
        "kind": "manifested_tree",
        "digest_sha256": digest,
        "filesystem_id": str(entry.stat().st_dev),
    }
    with patch.object(worker, "FLEET_WORKER_CACHE_ROOT", root):
        worker._verify_cache_filesystem(entry, binding, directory=True)

    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    alias_entry = alias / "sha256" / digest[:2] / digest
    alias_marker = {**marker, "cache_root": str(alias)}
    os.setxattr(
        entry,
        "user.fleet_compute_cache",
        (json.dumps(alias_marker, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    with patch.object(worker, "FLEET_WORKER_CACHE_ROOT", alias), pytest.raises(
        QwenRuntimeError
    ):
        worker._verify_cache_filesystem(alias_entry, binding, directory=True)


def test_descriptor_inspection_does_not_rehash_canonical_model() -> None:
    descriptor = _model_descriptor()
    backend = cache.AeonQwenArtifactBackend()
    with patch.object(cache, "load_artifact_identity") as verify, patch.object(
        backend, "_remote_python", return_value={"state": "absent"}
    ):
        assert (
            backend.inspect_entry(
                host="192.168.0.180",
                path=str(
                    fleet.FLEET_WORKER_CACHE_ROOT
                    / "sha256"
                    / descriptor.digest_sha256[:2]
                    / descriptor.digest_sha256
                ),
                descriptor=descriptor,
                expected_filesystem_id="123",
                verify_content=True,
            )
            is None
        )
    verify.assert_not_called()
