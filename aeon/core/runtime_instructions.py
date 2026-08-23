"""Private runtime handoff for per-instance instruction layers.

Nexus persists editable instruction bodies in its private registry.  A launcher
uses :func:`materialize_runtime_instructions` to publish one exact snapshot to
an owner-only file and passes only that file's path to Aeon.  Prompt text never
belongs in process arguments or environment values.

The reader deliberately treats a configured-but-invalid file as fatal.  Running
without ``AEON_INSTANCE_INSTRUCTIONS_FILE`` remains the normal standalone Aeon
behavior and supplies no additional instruction layers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


RUNTIME_INSTRUCTIONS_ENV = "AEON_INSTANCE_INSTRUCTIONS_FILE"
RUNTIME_INSTRUCTIONS_FILENAME = "runtime-instructions.json"
PROVIDER_INSTRUCTIONS_FILENAME = "provider-instructions.txt"
GROK_AGENT_PROFILE_FILENAME = "grok-agent-profile.md"
RUNTIME_INSTRUCTIONS_SCHEMA = 1
MAX_INSTRUCTION_LAYER_BYTES = 64 * 1024
# JSON may expand permitted control characters to six-byte ``\uXXXX`` escapes.
MAX_RUNTIME_INSTRUCTIONS_BYTES = MAX_INSTRUCTION_LAYER_BYTES * 12 + 32 * 1024
MAX_PROVIDER_INSTRUCTIONS_BYTES = MAX_INSTRUCTION_LAYER_BYTES * 2 + 4 * 1024
MAX_GROK_AGENT_PROFILE_BYTES = MAX_PROVIDER_INSTRUCTIONS_BYTES + 1024

_AGENT_KINDS = frozenset({"aeon", "codex", "claude", "grok"})
_PROVIDER_KINDS = frozenset({"codex", "claude", "grok"})
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeInstructionError(RuntimeError):
    """A private instruction snapshot could not be safely written or read."""


@dataclass(frozen=True)
class RuntimeInstructionLayers:
    """Validated runtime layers; content is excluded from representations."""

    instance_id: str = ""
    agent_kind: str = "aeon"
    profile_version_id: str | None = None
    profile_content: str = field(default="", repr=False)
    profile_content_sha256: str = field(default="", repr=False)
    local_revision: int = 0
    local_content: str = field(default="", repr=False)
    local_content_sha256: str = field(default="", repr=False)
    source_path: Path | None = None

    @property
    def is_empty(self) -> bool:
        return not self.profile_content and not self.local_content

    @property
    def applied_identity(self) -> dict[str, str | int | None]:
        """Arguments suitable for ``InstructionProfileService.mark_applied``."""

        return {
            "profile_version_id": self.profile_version_id,
            "local_revision": self.local_revision,
        }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_identifier(value: object, *, label: str, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise RuntimeInstructionError(f"Runtime instruction {label} is invalid")
    return value


def _layer_text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeInstructionError(f"Runtime instruction {label} is invalid")
    if "\x00" in value or len(value.encode("utf-8")) > MAX_INSTRUCTION_LAYER_BYTES:
        raise RuntimeInstructionError(f"Runtime instruction {label} is invalid")
    return value


def _digest(value: object, *, content: str, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise RuntimeInstructionError(f"Runtime instruction {label} digest is invalid")
    actual = _sha256_text(content)
    if not hmac.compare_digest(value, actual):
        raise RuntimeInstructionError(f"Runtime instruction {label} digest does not match")
    return value


def _revision(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**63 - 1
    ):
        raise RuntimeInstructionError("Runtime instruction local revision is invalid")
    return value


def _layers_from_mapping(payload: Mapping[str, object], *, source_path: Path | None) -> RuntimeInstructionLayers:
    schema = payload.get("schema_version", RUNTIME_INSTRUCTIONS_SCHEMA)
    if isinstance(schema, bool) or schema != RUNTIME_INSTRUCTIONS_SCHEMA:
        raise RuntimeInstructionError("Runtime instruction schema is unsupported")
    instance_id = _safe_identifier(payload.get("instance_id"), label="instance ID")
    agent_kind = payload.get("agent_kind")
    if not isinstance(agent_kind, str) or agent_kind not in _AGENT_KINDS:
        raise RuntimeInstructionError("Runtime instruction agent kind is invalid")
    profile_version_id = _safe_identifier(
        payload.get("profile_version_id"), label="profile version ID", allow_none=True
    )
    profile_content = _layer_text(payload.get("profile_content"), label="profile layer")
    profile_digest = _digest(
        payload.get("profile_content_sha256"),
        content=profile_content,
        label="profile layer",
    )
    local_revision = _revision(payload.get("local_revision"))
    local_content = _layer_text(payload.get("local_content"), label="local layer")
    local_digest = _digest(
        payload.get("local_content_sha256"), content=local_content, label="local layer"
    )
    if profile_version_id is None and profile_content:
        raise RuntimeInstructionError("Runtime instruction profile identity is missing")
    if local_revision == 0 and local_content:
        raise RuntimeInstructionError("Runtime instruction local identity is missing")
    return RuntimeInstructionLayers(
        instance_id=instance_id or "",
        agent_kind=agent_kind,
        profile_version_id=profile_version_id,
        profile_content=profile_content,
        profile_content_sha256=profile_digest,
        local_revision=local_revision,
        local_content=local_content,
        local_content_sha256=local_digest,
        source_path=source_path,
    )


def _payload_from_snapshot(snapshot: Mapping[str, object]) -> tuple[dict[str, object], RuntimeInstructionLayers]:
    if not isinstance(snapshot, Mapping):
        raise RuntimeInstructionError("Runtime instruction snapshot is invalid")
    candidate = {
        "schema_version": RUNTIME_INSTRUCTIONS_SCHEMA,
        "instance_id": snapshot.get("instance_id"),
        "agent_kind": snapshot.get("agent_kind"),
        "profile_version_id": snapshot.get("profile_version_id"),
        "profile_content": snapshot.get("profile_content"),
        "profile_content_sha256": snapshot.get("profile_content_sha256"),
        "local_revision": snapshot.get("local_revision"),
        "local_content": snapshot.get("local_content"),
        "local_content_sha256": snapshot.get("local_content_sha256"),
    }
    layers = _layers_from_mapping(candidate, source_path=None)
    return candidate, layers


def runtime_instruction_layers_from_snapshot(
    snapshot: Mapping[str, object],
) -> RuntimeInstructionLayers:
    """Validate a registry launch snapshot without exposing its bodies in repr."""

    _payload, layers = _payload_from_snapshot(snapshot)
    return layers


def _normalized_no_symlink_path(value: str | Path, *, label: str) -> Path:
    try:
        rendered = os.fspath(value)
    except TypeError as exc:
        raise RuntimeInstructionError(f"Runtime instruction {label} is invalid") from exc
    if not isinstance(rendered, str) or not rendered or "\x00" in rendered:
        raise RuntimeInstructionError(f"Runtime instruction {label} is invalid")
    absolute = Path(os.path.abspath(os.path.expanduser(rendered)))
    try:
        resolved = absolute.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RuntimeInstructionError(f"Runtime instruction {label} is unavailable") from exc
    if resolved != absolute:
        raise RuntimeInstructionError(f"Runtime instruction {label} contains a symbolic link")
    return absolute


def _validate_private_directory_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeInstructionError("Runtime instruction directory is not a directory")
    if metadata.st_uid != os.geteuid():
        raise RuntimeInstructionError("Runtime instruction directory has the wrong owner")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RuntimeInstructionError("Runtime instruction directory must have mode 0700")


def _private_directory(path: Path, *, create: bool) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return_descriptor = os.open(path, flags)
    except FileNotFoundError:
        if not create:
            raise RuntimeInstructionError("Runtime instruction directory is unavailable")
        parent = _normalized_no_symlink_path(path.parent, label="parent directory")
        try:
            parent_descriptor = os.open(parent, flags)
        except OSError as exc:
            raise RuntimeInstructionError("Runtime instruction parent directory is unavailable") from exc
        try:
            _validate_private_directory_metadata(os.fstat(parent_descriptor))
            try:
                os.mkdir(path.name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                pass
        finally:
            os.close(parent_descriptor)
        try:
            return_descriptor = os.open(path, flags)
        except OSError as exc:
            raise RuntimeInstructionError("Runtime instruction directory is unavailable") from exc
    except OSError as exc:
        raise RuntimeInstructionError("Runtime instruction directory is unavailable") from exc
    try:
        _validate_private_directory_metadata(os.fstat(return_descriptor))
    except Exception:
        os.close(return_descriptor)
        raise
    return return_descriptor


def _validate_private_file_metadata(
    metadata: os.stat_result, *, maximum: int = MAX_RUNTIME_INSTRUCTIONS_BYTES
) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeInstructionError("Runtime instruction file is not a regular file")
    if metadata.st_uid != os.geteuid():
        raise RuntimeInstructionError("Runtime instruction file has the wrong owner")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise RuntimeInstructionError("Runtime instruction file must have mode 0600")
    if metadata.st_size > maximum:
        raise RuntimeInstructionError("Runtime instruction file is too large")


def _publish_private_bytes(
    content: bytes,
    *,
    instance_dir: str | Path,
    filename: str,
    maximum: int,
) -> Path:
    if len(content) > maximum:
        raise RuntimeInstructionError("Runtime instruction snapshot is too large")

    directory = _normalized_no_symlink_path(instance_dir, label="directory")
    directory_descriptor = _private_directory(directory, create=True)
    temporary_name = f".{filename}.{secrets.token_hex(12)}.tmp"
    temporary_descriptor: int | None = None
    try:
        try:
            existing = os.stat(
                filename,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise RuntimeInstructionError("Runtime instruction target is unavailable") from exc
        if existing is not None:
            _validate_private_file_metadata(existing, maximum=maximum)

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_descriptor = os.open(
            temporary_name, flags, 0o600, dir_fd=directory_descriptor
        )
        os.fchmod(temporary_descriptor, 0o600)
        view = memoryview(content)
        while view:
            written = os.write(temporary_descriptor, view)
            if written <= 0:
                raise RuntimeInstructionError("Runtime instruction snapshot write failed")
            view = view[written:]
        os.fsync(temporary_descriptor)
        _validate_private_file_metadata(
            os.fstat(temporary_descriptor), maximum=maximum
        )
        os.close(temporary_descriptor)
        temporary_descriptor = None
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    except RuntimeInstructionError:
        raise
    except OSError as exc:
        raise RuntimeInstructionError("Runtime instruction snapshot could not be published") from exc
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        os.close(directory_descriptor)

    return directory / filename


def materialize_runtime_instructions(
    snapshot: Mapping[str, object], instance_dir: str | Path
) -> Path:
    """Atomically publish one validated launch snapshot beneath ``instance_dir``.

    ``instance_dir`` may already exist or may be created directly beneath an
    existing owner-private directory.  Existing symlinks, non-regular targets,
    wrong owners, and permissive modes are rejected rather than repaired.
    """

    payload, _layers = _payload_from_snapshot(snapshot)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _publish_private_bytes(
        encoded,
        instance_dir=instance_dir,
        filename=RUNTIME_INSTRUCTIONS_FILENAME,
        maximum=MAX_RUNTIME_INSTRUCTIONS_BYTES,
    )


def _read_bounded(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    remaining = MAX_RUNTIME_INSTRUCTIONS_BYTES + 1
    while remaining:
        chunk = os.read(descriptor, min(remaining, 65536))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    result = b"".join(chunks)
    if len(result) > MAX_RUNTIME_INSTRUCTIONS_BYTES:
        raise RuntimeInstructionError("Runtime instruction file is too large")
    return result


def load_runtime_instructions(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    expected_instance_id: str | None = None,
    expected_agent_kind: str | None = None,
) -> RuntimeInstructionLayers:
    """Load and authenticate a private runtime snapshot.

    When ``path`` is omitted, only :data:`RUNTIME_INSTRUCTIONS_ENV` is read from
    the environment.  An absent/empty setting returns an empty layer set; a
    present but unsafe or malformed setting raises :class:`RuntimeInstructionError`.
    """

    if path is None:
        source_environment = os.environ if environ is None else environ
        configured = source_environment.get(RUNTIME_INSTRUCTIONS_ENV, "")
        if not configured:
            empty_digest = _sha256_text("")
            return RuntimeInstructionLayers(
                instance_id="standalone",
                profile_content_sha256=empty_digest,
                local_content_sha256=empty_digest,
            )
        path = configured

    source_path = _normalized_no_symlink_path(path, label="file path")
    directory_descriptor = _private_directory(source_path.parent, create=False)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(source_path.name, flags, dir_fd=directory_descriptor)
        except OSError as exc:
            raise RuntimeInstructionError("Runtime instruction file is unavailable") from exc
        _validate_private_file_metadata(os.fstat(descriptor))
        raw = _read_bounded(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_descriptor)

    try:
        decoded = raw.decode("utf-8")
        payload = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeInstructionError("Runtime instruction file is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeInstructionError("Runtime instruction document is invalid")
    expected_keys = {
        "schema_version",
        "instance_id",
        "agent_kind",
        "profile_version_id",
        "profile_content",
        "profile_content_sha256",
        "local_revision",
        "local_content",
        "local_content_sha256",
    }
    if set(payload) != expected_keys:
        raise RuntimeInstructionError("Runtime instruction document fields are invalid")
    layers = _layers_from_mapping(payload, source_path=source_path)
    if expected_instance_id is not None:
        expected_id = _safe_identifier(expected_instance_id, label="expected instance ID")
        if not hmac.compare_digest(layers.instance_id, expected_id or ""):
            raise RuntimeInstructionError("Runtime instructions belong to another instance")
    if expected_agent_kind is not None:
        if expected_agent_kind not in _AGENT_KINDS:
            raise RuntimeInstructionError("Expected runtime instruction agent kind is invalid")
        if not hmac.compare_digest(layers.agent_kind, expected_agent_kind):
            raise RuntimeInstructionError("Runtime instructions belong to another agent kind")
    return layers


def _revalidate_layers(layers: RuntimeInstructionLayers) -> RuntimeInstructionLayers:
    if not isinstance(layers, RuntimeInstructionLayers):
        raise RuntimeInstructionError("Runtime instruction layers are invalid")
    payload = {
        "schema_version": RUNTIME_INSTRUCTIONS_SCHEMA,
        "instance_id": layers.instance_id,
        "agent_kind": layers.agent_kind,
        "profile_version_id": layers.profile_version_id,
        "profile_content": layers.profile_content,
        "profile_content_sha256": layers.profile_content_sha256,
        "local_revision": layers.local_revision,
        "local_content": layers.local_content,
        "local_content_sha256": layers.local_content_sha256,
    }
    return _layers_from_mapping(payload, source_path=layers.source_path)


def format_runtime_instruction_layers(layers: RuntimeInstructionLayers) -> str:
    """Render exact bodies as provider-neutral, truthfully labelled sections."""

    validated = _revalidate_layers(layers)
    sections: list[str] = []
    if validated.profile_content:
        sections.append(
            "**NEXUS LOCALLY KNOWN INSTRUCTION PROFILE**\n"
            "This is a locally selected instruction layer, not a vendor-hidden system prompt.\n"
            "--- BEGIN NEXUS PROFILE LAYER ---\n"
            f"{validated.profile_content}\n"
            "--- END NEXUS PROFILE LAYER ---"
        )
    if validated.local_content:
        sections.append(
            "**NEXUS INSTANCE-SPECIFIC ROLE**\n"
            "This layer applies only to this managed agent instance.\n"
            "--- BEGIN NEXUS LOCAL ROLE LAYER ---\n"
            f"{validated.local_content}\n"
            "--- END NEXUS LOCAL ROLE LAYER ---"
        )
    return "\n\n".join(sections)


def materialize_provider_instruction_text(
    snapshot: Mapping[str, object], instance_dir: str | Path
) -> Path:
    """Publish one provider overlay as an atomic owner-private UTF-8 text file."""

    layers = runtime_instruction_layers_from_snapshot(snapshot)
    if layers.agent_kind not in _PROVIDER_KINDS:
        raise RuntimeInstructionError(
            "Provider instruction text requires a provider agent snapshot"
        )
    content = format_runtime_instruction_layers(layers).encode("utf-8")
    return _publish_private_bytes(
        content,
        instance_dir=instance_dir,
        filename=PROVIDER_INSTRUCTIONS_FILENAME,
        maximum=MAX_PROVIDER_INSTRUCTIONS_BYTES,
    )


def materialize_grok_agent_profile(
    snapshot: Mapping[str, object], instance_dir: str | Path
) -> Path:
    """Publish a private Grok ``--agent`` definition with an appended prompt body.

    Grok agent definitions require only a bounded YAML frontmatter identity; the
    Markdown body is interpreted in the default ``extend`` prompt mode.  Passing
    this owner-only file path keeps the instruction bodies out of argv while
    retaining Grok's normal tools, permissions, and workspace AGENTS.md loading.
    """

    layers = runtime_instruction_layers_from_snapshot(snapshot)
    if layers.agent_kind != "grok":
        raise RuntimeInstructionError("Grok agent profile requires a Grok snapshot")
    rendered = format_runtime_instruction_layers(layers)
    profile_name = "nexus-" + hashlib.sha256(
        layers.instance_id.encode("utf-8")
    ).hexdigest()[:20]
    content = (
        "---\n"
        f"name: {profile_name}\n"
        "description: Nexus-managed persistent instruction overlay\n"
        "---\n\n"
        f"{rendered}\n"
    ).encode("utf-8")
    return _publish_private_bytes(
        content,
        instance_dir=instance_dir,
        filename=GROK_AGENT_PROFILE_FILENAME,
        maximum=MAX_GROK_AGENT_PROFILE_BYTES,
    )


def format_aeon_runtime_instructions(layers: RuntimeInstructionLayers) -> str:
    """Render the generic instruction layers for inclusion in an Aeon prompt."""

    if layers.agent_kind != "aeon":
        raise RuntimeInstructionError("Runtime instructions do not belong to an Aeon instance")
    rendered = format_runtime_instruction_layers(layers)
    return f"\n\n{rendered}" if rendered else ""
