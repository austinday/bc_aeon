"""Hermetic tests for the pinned modular OpenCode harness foundation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
from types import SimpleNamespace
from pathlib import Path

import pytest

from aeon.harnesses.catalog import (
    DEFAULT_HARNESS_ID,
    LEGACY_AEON_HARNESS_ID,
    OPENCODE_ARCHIVE_SHA256,
    OPENCODE_EXECUTABLE_SHA256,
    OPENCODE_HARNESS_ID,
    OPENCODE_VERSION,
    HarnessArtifact,
    normalize_harness_id,
    public_harness_catalog,
)
from aeon.harnesses.opencode_install import (
    INSTALL_RECEIPT_NAME,
    OpenCodeInstallError,
    install_opencode,
    main,
    opencode_binary_path,
    opencode_status,
    resolve_opencode_binary,
    resolve_opencode_home,
    _probe_version,
)


def _fake_archive(
    directory: Path,
    *,
    version: str = OPENCODE_VERSION,
    member_kind: str = "file",
) -> tuple[Path, HarnessArtifact]:
    archive_path = directory / "opencode-linux-x64.tar.gz"
    script = (
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        f"  printf '%s\\n' '{version}'\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n"
    ).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("opencode")
        member.mode = 0o755
        if member_kind == "file":
            member.size = len(script)
            archive.addfile(member, io.BytesIO(script))
        elif member_kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            archive.addfile(member)
        else:  # pragma: no cover - helper misuse
            raise AssertionError(member_kind)
    archive_path.chmod(0o600)
    payload = archive_path.read_bytes()
    artifact = HarnessArtifact(
        version=OPENCODE_VERSION,
        system="Linux",
        machines=("x86_64", "amd64"),
        archive_name=archive_path.name,
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        archive_size=len(payload),
        executable_sha256=hashlib.sha256(script).hexdigest(),
        executable_size=len(script),
        url=(
            "https://github.com/anomalyco/opencode/releases/download/"
            f"v{OPENCODE_VERSION}/{archive_path.name}"
        ),
        executable_name="opencode",
    )
    return archive_path, artifact


class _FakeDownload(io.BytesIO):
    def __init__(self, payload: bytes, url: str):
        super().__init__(payload)
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_catalog_has_strict_stable_ids_and_official_pin() -> None:
    assert DEFAULT_HARNESS_ID == OPENCODE_HARNESS_ID == "opencode"
    assert LEGACY_AEON_HARNESS_ID == "legacy-aeon"
    assert OPENCODE_VERSION == "1.18.27"
    assert OPENCODE_ARCHIVE_SHA256 == (
        "4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
    )
    assert OPENCODE_EXECUTABLE_SHA256 == (
        "bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256"
    )
    assert normalize_harness_id(None) == "opencode"
    assert normalize_harness_id("") == "opencode"
    assert normalize_harness_id(" OpenCode ") == "opencode"
    assert normalize_harness_id("LEGACY-AEON") == "legacy-aeon"
    with pytest.raises(ValueError, match="unsupported"):
        normalize_harness_id("aeon")
    with pytest.raises(ValueError, match="string"):
        normalize_harness_id(7)

    catalog = public_harness_catalog()
    assert json.loads(json.dumps(catalog)) == catalog
    assert [item["id"] for item in catalog] == ["opencode", "legacy-aeon"]
    assert [item["id"] for item in catalog if item["default"]] == ["opencode"]


def test_status_is_side_effect_free_when_home_is_missing(tmp_path: Path) -> None:
    home = tmp_path / "does-not-exist" / "opencode"
    before = list(tmp_path.iterdir())
    result = opencode_status(home, system="Linux", machine="x86_64")
    assert result["state"] == "missing"
    assert result["ready"] is False
    assert result["installed"] is False
    assert list(tmp_path.iterdir()) == before
    assert not home.exists()


def test_version_probe_uses_minimal_non_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    def fake_run(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="1.18.27\n", stderr="secret")

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("GPU_AGENT_CLAIM_ID", "must-not-leak")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/must-not-load.so")
    monkeypatch.setenv("OPENCODE_CONFIG", "/tmp/must-not-use.json")
    monkeypatch.setattr("aeon.harnesses.opencode_install.subprocess.run", fake_run)

    assert _probe_version(tmp_path / "opencode") == "1.18.27"
    environment = observed["kwargs"]["env"]
    assert environment["OPENCODE_DISABLE_AUTOUPDATE"] == "true"
    assert environment["OPENCODE_DISABLE_MODELS_FETCH"] == "true"
    assert environment["OPENCODE_DISABLE_PROJECT_CONFIG"] == "true"
    assert environment["OPENCODE_DISABLE_DEFAULT_PLUGINS"] == "true"
    for name in (
        "OPENAI_API_KEY",
        "GPU_AGENT_CLAIM_ID",
        "LD_PRELOAD",
        "OPENCODE_CONFIG",
    ):
        assert name not in environment
    def failed_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="token=must-not-leak",
        )

    monkeypatch.setattr(
        "aeon.harnesses.opencode_install.subprocess.run", failed_run
    )
    with pytest.raises(OpenCodeInstallError) as failure:
        _probe_version(tmp_path / "opencode")
    assert "must-not-leak" not in str(failure.value)


def test_resolver_never_falls_back_to_path(tmp_path: Path, monkeypatch) -> None:
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    impostor = path_bin / "opencode"
    impostor.write_text("#!/bin/sh\nprintf '1.18.27\\n'\n", encoding="utf-8")
    impostor.chmod(0o700)
    monkeypatch.setenv("PATH", str(path_bin))
    with pytest.raises(OpenCodeInstallError, match="does not exist"):
        resolve_opencode_binary(tmp_path / "missing-home")


def test_home_must_be_absolute_and_environment_is_supported(tmp_path: Path) -> None:
    with pytest.raises(OpenCodeInstallError, match="absolute"):
        resolve_opencode_home("relative/opencode")
    assert resolve_opencode_home(environ={"AEON_OPENCODE_HOME": str(tmp_path)}) == tmp_path


def test_install_is_private_atomic_idempotent_and_resolvable(tmp_path: Path) -> None:
    archive, artifact = _fake_archive(tmp_path)
    home = tmp_path / "state" / "opencode"

    installed = install_opencode(
        home,
        archive_path=archive,
        artifact=artifact,
        system="Linux",
        machine="x86_64",
    )
    assert installed.ready is True
    assert installed.state == "ready"
    binary = Path(installed.binary)
    version_dir = binary.parent
    receipt_path = version_dir / INSTALL_RECEIPT_NAME
    assert binary == opencode_binary_path(home, artifact=artifact)
    assert binary.read_bytes().startswith(b"#!/bin/sh")
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert stat.S_IMODE((home / "versions").stat().st_mode) == 0o700
    assert stat.S_IMODE(version_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(binary.stat().st_mode) == 0o700
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert not list((home / "versions").glob(".install-*"))

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["archive_sha256"] == artifact.archive_sha256
    assert receipt["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert receipt["version"] == artifact.version
    assert resolve_opencode_binary(home, artifact=artifact) == binary

    inode = binary.stat().st_ino
    second = install_opencode(
        home,
        archive_path=archive,
        artifact=artifact,
        system="Linux",
        machine="x86_64",
    )
    assert second.ready is True
    assert binary.stat().st_ino == inode


def test_wrong_platform_is_rejected_before_creating_home(tmp_path: Path) -> None:
    archive, artifact = _fake_archive(tmp_path)
    home = tmp_path / "opencode"
    with pytest.raises(OpenCodeInstallError, match="Linux x86-64"):
        install_opencode(
            home,
            archive_path=archive,
            artifact=artifact,
            system="Darwin",
            machine="arm64",
        )
    assert not home.exists()
    status = opencode_status(
        home, artifact=artifact, system="Darwin", machine="arm64"
    )
    assert status["state"] == "unsupported"


def test_digest_mismatch_never_publishes_a_version(tmp_path: Path) -> None:
    archive, artifact = _fake_archive(tmp_path)
    bad_artifact = HarnessArtifact(
        **{**artifact.__dict__, "archive_sha256": "0" * 64}
    )
    home = tmp_path / "opencode"
    with pytest.raises(OpenCodeInstallError, match="digest"):
        install_opencode(
            home,
            archive_path=archive,
            artifact=bad_artifact,
            system="Linux",
            machine="x86_64",
        )
    assert not opencode_binary_path(home, artifact=bad_artifact).parent.exists()
    assert not list((home / "versions").glob(".install-*"))


def test_download_path_uses_only_the_pinned_payload_and_approved_origin(
    tmp_path: Path,
) -> None:
    archive, artifact = _fake_archive(tmp_path)
    payload = archive.read_bytes()
    seen = []

    def opener(request, *, timeout):
        seen.append((request.full_url, timeout))
        return _FakeDownload(payload, artifact.url)

    home = tmp_path / "downloaded"
    result = install_opencode(
        home,
        artifact=artifact,
        opener=opener,
        system="Linux",
        machine="x86_64",
    )
    assert result.ready is True
    assert seen == [(artifact.url, 60)]
    assert not list(result.binary and Path(result.binary).parent.glob("*.tar.gz"))


def test_download_redirect_to_unapproved_origin_never_publishes(tmp_path: Path) -> None:
    archive, artifact = _fake_archive(tmp_path)
    payload = archive.read_bytes()

    def opener(_request, *, timeout):
        assert timeout == 60
        return _FakeDownload(payload, "https://attacker.invalid/opencode.tar.gz")

    home = tmp_path / "redirected"
    with pytest.raises(OpenCodeInstallError, match="unapproved origin"):
        install_opencode(
            home,
            artifact=artifact,
            opener=opener,
            system="Linux",
            machine="x86_64",
        )
    assert not opencode_binary_path(home, artifact=artifact).parent.exists()
    assert not list((home / "versions").glob(".install-*"))


def test_wrong_reported_version_never_publishes(tmp_path: Path) -> None:
    archive, artifact = _fake_archive(tmp_path, version="1.18.26")
    home = tmp_path / "opencode"
    with pytest.raises(OpenCodeInstallError, match="pinned version"):
        install_opencode(
            home,
            archive_path=archive,
            artifact=artifact,
            system="Linux",
            machine="x86_64",
        )
    assert not opencode_binary_path(home, artifact=artifact).parent.exists()


def test_archive_symlink_member_is_rejected_without_escape(tmp_path: Path) -> None:
    archive, artifact = _fake_archive(tmp_path, member_kind="symlink")
    home = tmp_path / "opencode"
    outside = tmp_path / "outside"
    with pytest.raises(OpenCodeInstallError, match="unsafe"):
        install_opencode(
            home,
            archive_path=archive,
            artifact=artifact,
            system="Linux",
            machine="x86_64",
        )
    assert not outside.exists()
    assert not opencode_binary_path(home, artifact=artifact).parent.exists()


def test_status_rejects_tamper_permissions_hardlinks_and_symlinks(tmp_path: Path) -> None:
    archive, artifact = _fake_archive(tmp_path)

    def fresh(name: str) -> Path:
        home = tmp_path / name
        install_opencode(
            home,
            archive_path=archive,
            artifact=artifact,
            system="Linux",
            machine="x86_64",
        )
        return home

    tampered = fresh("tampered")
    binary = opencode_binary_path(tampered, artifact=artifact)
    binary.write_bytes(binary.read_bytes() + b"# changed\n")
    assert opencode_status(
        tampered, artifact=artifact, system="Linux", machine="x86_64"
    )["state"] == "invalid"

    permissive = fresh("permissive")
    opencode_binary_path(permissive, artifact=artifact).chmod(0o755)
    result = opencode_status(
        permissive, artifact=artifact, system="Linux", machine="x86_64"
    )
    assert result["state"] == "invalid"
    assert "non-private" in result["reason"]

    linked = fresh("hardlinked")
    linked_binary = opencode_binary_path(linked, artifact=artifact)
    os.link(linked_binary, tmp_path / "second-link")
    result = opencode_status(
        linked, artifact=artifact, system="Linux", machine="x86_64"
    )
    assert result["state"] == "invalid"
    assert "multiply-linked" in result["reason"]

    symlink_home = tmp_path / "symlink-home"
    symlink_home.symlink_to(fresh("real-home"), target_is_directory=True)
    result = opencode_status(
        symlink_home, artifact=artifact, system="Linux", machine="x86_64"
    )
    assert result["state"] == "invalid"
    assert "symlink" in result["reason"]


def test_invalid_existing_version_is_never_replaced(tmp_path: Path) -> None:
    archive, artifact = _fake_archive(tmp_path)
    home = tmp_path / "opencode"
    version_dir = opencode_binary_path(home, artifact=artifact).parent
    version_dir.mkdir(parents=True, mode=0o700)
    home.chmod(0o700)
    (home / "versions").chmod(0o700)
    marker = version_dir / "operator-data"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(OpenCodeInstallError, match="refusing to replace"):
        install_opencode(
            home,
            archive_path=archive,
            artifact=artifact,
            system="Linux",
            machine="x86_64",
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_status_cli_reports_missing_without_installing(tmp_path: Path, capsys) -> None:
    home = tmp_path / "opencode"
    assert main(["status", "--home", str(home), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "missing"
    assert payload["ready"] is False
    assert not home.exists()
