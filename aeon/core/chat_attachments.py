"""Private, bounded media storage for the Nexus main-orchestrator chat.

Attachment bytes never enter presence telemetry or the Nexus database.  They
live beside the owning chat transcript in its mode-0700 instance directory and
are exposed to the browser only through an authenticated, transcript-bound
download endpoint.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from PIL import Image


ATTACHMENT_DIRECTORY = "chat-attachments"
ATTACHMENT_LOCK_FILENAME = "chat-attachments.lock"
MAX_ATTACHMENTS_PER_MESSAGE = 4
MAX_ATTACHMENT_BYTES = 128 * 1024 * 1024
MAX_ATTACHMENT_REQUEST_BYTES = 256 * 1024 * 1024
MAX_ATTACHMENT_STORAGE_BYTES = 2 * 1024 * 1024 * 1024
_ATTACHMENT_ID_RE = re.compile(r"^att-[0-9a-f]{32}$")
_MIME_DETAILS = {
    "image/jpeg": ("image", ".jpg"),
    "image/png": ("image", ".png"),
    "image/webp": ("image", ".webp"),
    "image/gif": ("image", ".gif"),
    "video/mp4": ("video", ".mp4"),
    "video/quicktime": ("video", ".mov"),
    "video/webm": ("video", ".webm"),
    "audio/mpeg": ("audio", ".mp3"),
    "audio/wav": ("audio", ".wav"),
    "audio/x-wav": ("audio", ".wav"),
    "audio/ogg": ("audio", ".ogg"),
    "audio/flac": ("audio", ".flac"),
    "audio/mp4": ("audio", ".m4a"),
    "audio/x-m4a": ("audio", ".m4a"),
}
_GENERATED_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}


class ChatAttachmentError(RuntimeError):
    """An uploaded attachment failed a private-storage or media check."""


@dataclass(frozen=True)
class StoredChatAttachment:
    attachment_id: str
    name: str
    media_type: str
    mime_type: str
    size_bytes: int
    path: Path

    def public(self) -> dict[str, object]:
        return {
            "id": self.attachment_id,
            "name": self.name,
            "media_type": self.media_type,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class _GeneratedMediaUpload:
    """Upload-shaped wrapper around an already-open, verified local artifact."""

    filename: str
    content_type: str
    file: BinaryIO


def _owned_private_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ChatAttachmentError("Chat attachment storage is unavailable") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise ChatAttachmentError("Chat attachment storage is not owner-private")
    return descriptor


def _attachment_directory(instance_directory: str | Path) -> Path:
    instance = Path(instance_directory)
    if not instance.is_absolute():
        raise ChatAttachmentError("Chat attachment storage path is invalid")
    parent_fd = _owned_private_directory(instance)
    try:
        try:
            os.mkdir(ATTACHMENT_DIRECTORY, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    except OSError as exc:
        raise ChatAttachmentError("Chat attachment storage is unavailable") from exc
    finally:
        os.close(parent_fd)
    directory = instance / ATTACHMENT_DIRECTORY
    child_fd = _owned_private_directory(directory)
    os.close(child_fd)
    return directory


def _safe_display_name(value: object) -> str:
    rendered = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    rendered = "".join(
        character
        for character in rendered
        if ord(character) >= 32 and ord(character) != 127
    ).strip()
    if not rendered:
        rendered = "attachment"
    encoded = rendered.encode("utf-8")
    while len(encoded) > 180:
        rendered = rendered[:-1]
        encoded = rendered.encode("utf-8")
    return rendered or "attachment"


def _declared_mime(value: object) -> str:
    mime = str(value or "").split(";", 1)[0].strip().lower()
    if mime not in _MIME_DETAILS:
        raise ChatAttachmentError(
            "Use a JPEG, PNG, WebP, GIF, MP4, MOV, WebM, MP3, WAV, OGG, FLAC, or M4A file"
        )
    return mime


def _magic_matches(path: Path, mime_type: str) -> bool:
    with path.open("rb") as stream:
        header = stream.read(32)
    if mime_type == "image/jpeg":
        return header.startswith(b"\xff\xd8\xff")
    if mime_type == "image/png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/webp":
        return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    if mime_type == "image/gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    if mime_type in {"video/mp4", "video/quicktime", "audio/mp4", "audio/x-m4a"}:
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if mime_type == "video/webm":
        return header.startswith(b"\x1aE\xdf\xa3")
    if mime_type == "audio/mpeg":
        return header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        )
    if mime_type in {"audio/wav", "audio/x-wav"}:
        return header.startswith(b"RIFF") and header[8:12] == b"WAVE"
    if mime_type == "audio/ogg":
        return header.startswith(b"OggS")
    if mime_type == "audio/flac":
        return header.startswith(b"fLaC")
    return False


def _validate_media(path: Path, mime_type: str) -> None:
    if not _magic_matches(path, mime_type):
        raise ChatAttachmentError("The uploaded bytes do not match the declared media type")
    if not mime_type.startswith("image/"):
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
    except Exception as exc:
        raise ChatAttachmentError("The uploaded image is corrupt or unsafe to decode") from exc


def _regular_storage_bytes(directory: Path) -> int:
    total = 0
    for entry in directory.iterdir():
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise ChatAttachmentError("Chat attachment storage cannot be measured") from exc
        if entry.name == ATTACHMENT_LOCK_FILENAME:
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ChatAttachmentError("Chat attachment storage contains an unsafe entry")
        total += metadata.st_size
    return total


def _open_lock(directory: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(directory / ATTACHMENT_LOCK_FILENAME, flags, 0o600)
    except OSError as exc:
        raise ChatAttachmentError("Chat attachment storage is unavailable") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise ChatAttachmentError("Chat attachment lock is unsafe")
    return descriptor


def store_chat_attachments(
    instance_directory: str | Path,
    uploads: Iterable[object],
) -> list[StoredChatAttachment]:
    """Validate and atomically retain one bounded set of uploaded media files."""

    upload_list = list(uploads)
    if not upload_list:
        return []
    if len(upload_list) > MAX_ATTACHMENTS_PER_MESSAGE:
        raise ChatAttachmentError(
            f"Attach at most {MAX_ATTACHMENTS_PER_MESSAGE} files to one message"
        )
    directory = _attachment_directory(instance_directory)
    lock_fd = _open_lock(directory)
    created: list[StoredChatAttachment] = []
    temporary_paths: list[Path] = []
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current_bytes = _regular_storage_bytes(directory)
        request_bytes = 0
        for upload in upload_list:
            name = _safe_display_name(getattr(upload, "filename", ""))
            mime_type = _declared_mime(getattr(upload, "content_type", ""))
            media_type, extension = _MIME_DETAILS[mime_type]
            source: BinaryIO | None = getattr(upload, "file", None)
            if source is None or not callable(getattr(source, "read", None)):
                raise ChatAttachmentError("The uploaded media stream is invalid")
            try:
                source.seek(0)
            except (OSError, AttributeError):
                pass
            attachment_id = f"att-{uuid.uuid4().hex}"
            final_path = directory / f"{attachment_id}{extension}"
            temporary_path = directory / f".{attachment_id}.upload"
            temporary_paths.append(temporary_path)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary_path, flags, 0o600)
            size = 0
            try:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise ChatAttachmentError("The uploaded media stream is invalid")
                    size += len(chunk)
                    request_bytes += len(chunk)
                    if size > MAX_ATTACHMENT_BYTES:
                        raise ChatAttachmentError("One attachment exceeds the 128 MB limit")
                    if request_bytes > MAX_ATTACHMENT_REQUEST_BYTES:
                        raise ChatAttachmentError("The attachments exceed the 256 MB message limit")
                    if current_bytes + request_bytes > MAX_ATTACHMENT_STORAGE_BYTES:
                        raise ChatAttachmentError("This chat reached its 2 GB attachment limit")
                    view = memoryview(chunk)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise ChatAttachmentError("The attachment could not be saved")
                        view = view[written:]
                if size <= 0:
                    raise ChatAttachmentError("Empty attachments are not supported")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _validate_media(temporary_path, mime_type)
            os.replace(temporary_path, final_path)
            temporary_paths.remove(temporary_path)
            created.append(
                StoredChatAttachment(
                    attachment_id=attachment_id,
                    name=name,
                    media_type=media_type,
                    mime_type=mime_type,
                    size_bytes=size,
                    path=final_path,
                )
            )
        return created
    except BaseException:
        for attachment in created:
            try:
                attachment.path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def store_generated_chat_attachments(
    instance_directory: str | Path,
    source_paths: Iterable[str | Path],
) -> list[StoredChatAttachment]:
    """Copy exact, owner-held media artifacts into one chat's private storage.

    The caller supplies paths only after a successful media-tool receipt.  Each
    source is opened with ``O_NOFOLLOW`` and copied from that descriptor, so a
    model-returned path can never turn the authenticated attachment endpoint
    into a general filesystem reader.
    """

    paths = [Path(value) for value in source_paths]
    if not paths:
        return []
    if len(paths) > MAX_ATTACHMENTS_PER_MESSAGE:
        raise ChatAttachmentError(
            f"Attach at most {MAX_ATTACHMENTS_PER_MESSAGE} generated files to one message"
        )

    streams: list[BinaryIO] = []
    uploads: list[_GeneratedMediaUpload] = []
    try:
        for path in paths:
            if not path.is_absolute():
                raise ChatAttachmentError("Generated attachment paths must be absolute")
            mime_type = _GENERATED_MIME_BY_SUFFIX.get(path.suffix.lower())
            if mime_type is None:
                raise ChatAttachmentError("Generated attachment type is unsupported")
            flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise ChatAttachmentError("Generated attachment is unavailable") from exc
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or metadata.st_size <= 0
                    or metadata.st_size > MAX_ATTACHMENT_BYTES
                ):
                    raise ChatAttachmentError("Generated attachment is not safely owned")
                stream = os.fdopen(descriptor, "rb", closefd=True)
                descriptor = -1
                streams.append(stream)
                uploads.append(
                    _GeneratedMediaUpload(
                        filename=_safe_display_name(path.name),
                        content_type=mime_type,
                        file=stream,
                    )
                )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        return store_chat_attachments(instance_directory, uploads)
    finally:
        for stream in streams:
            stream.close()


def clone_chat_attachments(
    source_instance_directory: str | Path,
    target_instance_directory: str | Path,
    metadata_items: Iterable[dict[str, object]],
) -> list[StoredChatAttachment]:
    """Copy transcript-bound attachments into an isolated fork directory.

    Every source is resolved through the same owner/mode/size checks used by the
    authenticated download path. New attachment IDs prevent the fork from
    depending on the lifetime of its parent session.
    """

    metadata = list(metadata_items)
    if not metadata:
        return []
    if len(metadata) > MAX_ATTACHMENTS_PER_MESSAGE:
        raise ChatAttachmentError("Chat attachment metadata is invalid")
    streams: list[BinaryIO] = []
    uploads: list[_GeneratedMediaUpload] = []
    try:
        for item in metadata:
            if not isinstance(item, dict):
                raise ChatAttachmentError("Chat attachment metadata is invalid")
            source = resolve_chat_attachment(source_instance_directory, item)
            flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(source, flags)
            try:
                observed = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or observed.st_uid != os.geteuid()
                    or observed.st_nlink != 1
                    or observed.st_size != item.get("size_bytes")
                ):
                    raise ChatAttachmentError("Chat attachment is not safely owned")
                stream = os.fdopen(descriptor, "rb", closefd=True)
                descriptor = -1
                streams.append(stream)
                uploads.append(
                    _GeneratedMediaUpload(
                        filename=_safe_display_name(item.get("name")),
                        content_type=_declared_mime(item.get("mime_type")),
                        file=stream,
                    )
                )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        return store_chat_attachments(target_instance_directory, uploads)
    finally:
        for stream in streams:
            stream.close()


def remove_chat_attachments(attachments: Iterable[StoredChatAttachment]) -> None:
    """Rollback only exact files created by the current failed delivery."""

    for attachment in attachments:
        try:
            attachment.path.unlink()
        except FileNotFoundError:
            pass


def resolve_chat_attachment(
    instance_directory: str | Path,
    metadata: dict[str, object],
) -> Path:
    """Resolve one transcript-validated public record to its owned data file."""

    attachment_id = str(metadata.get("id") or "")
    mime_type = str(metadata.get("mime_type") or "")
    if not _ATTACHMENT_ID_RE.fullmatch(attachment_id) or mime_type not in _MIME_DETAILS:
        raise ChatAttachmentError("Chat attachment identity is invalid")
    directory = _attachment_directory(instance_directory)
    extension = _MIME_DETAILS[mime_type][1]
    path = directory / f"{attachment_id}{extension}"
    try:
        file_metadata = path.lstat()
    except OSError as exc:
        raise ChatAttachmentError("Chat attachment is unavailable") from exc
    if (
        not stat.S_ISREG(file_metadata.st_mode)
        or file_metadata.st_uid != os.geteuid()
        or file_metadata.st_nlink != 1
        or stat.S_IMODE(file_metadata.st_mode) != 0o600
        or file_metadata.st_size != metadata.get("size_bytes")
    ):
        raise ChatAttachmentError("Chat attachment is not safely owned")
    return path
