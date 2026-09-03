"""Stable identities and pinned artifacts for Aeon's modular harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


OPENCODE_HARNESS_ID = "opencode"
LEGACY_AEON_HARNESS_ID = "legacy-aeon"
DEFAULT_HARNESS_ID = OPENCODE_HARNESS_ID

OPENCODE_VERSION = "1.18.27"
OPENCODE_RELEASE_TAG = f"v{OPENCODE_VERSION}"
OPENCODE_PLATFORM = "linux-x64"
OPENCODE_ARCHIVE_NAME = f"opencode-{OPENCODE_PLATFORM}.tar.gz"
# GitHub's immutable v1.18.27 release metadata publishes the archive digest and
# size.  The executable digest/size below were measured from its sole member
# after independently verifying that exact archive pin.
OPENCODE_ARCHIVE_SHA256 = (
    "4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
)
OPENCODE_ARCHIVE_SIZE = 60_524_016
OPENCODE_EXECUTABLE_SHA256 = (
    "bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256"
)
OPENCODE_EXECUTABLE_SIZE = 184_633_472
OPENCODE_RELEASE_URL = (
    "https://github.com/anomalyco/opencode/releases/download/"
    f"{OPENCODE_RELEASE_TAG}/{OPENCODE_ARCHIVE_NAME}"
)
OPENCODE_EXECUTABLE_NAME = "opencode"
OPENCODE_HOME_ENV = "AEON_OPENCODE_HOME"


@dataclass(frozen=True)
class HarnessArtifact:
    """One immutable native artifact accepted by a harness installer."""

    version: str
    system: str
    machines: tuple[str, ...]
    archive_name: str
    archive_sha256: str
    archive_size: int
    executable_sha256: str
    executable_size: int
    url: str
    executable_name: str


@dataclass(frozen=True)
class HarnessDefinition:
    """Internal immutable definition for a selectable harness."""

    harness_id: str
    label: str
    description: str
    artifact: HarnessArtifact | None = None


OPENCODE_ARTIFACT = HarnessArtifact(
    version=OPENCODE_VERSION,
    system="Linux",
    machines=("x86_64", "amd64"),
    archive_name=OPENCODE_ARCHIVE_NAME,
    archive_sha256=OPENCODE_ARCHIVE_SHA256,
    archive_size=OPENCODE_ARCHIVE_SIZE,
    executable_sha256=OPENCODE_EXECUTABLE_SHA256,
    executable_size=OPENCODE_EXECUTABLE_SIZE,
    url=OPENCODE_RELEASE_URL,
    executable_name=OPENCODE_EXECUTABLE_NAME,
)

HARNESS_CATALOG: Mapping[str, HarnessDefinition] = MappingProxyType(
    {
        OPENCODE_HARNESS_ID: HarnessDefinition(
            harness_id=OPENCODE_HARNESS_ID,
            label="OpenCode",
            description="Pinned OpenCode harness with Nexus local tools and models.",
            artifact=OPENCODE_ARTIFACT,
        ),
        LEGACY_AEON_HARNESS_ID: HarnessDefinition(
            harness_id=LEGACY_AEON_HARNESS_ID,
            label="Aeon (legacy)",
            description="Original Aeon reasoning-loop harness.",
        ),
    }
)


def normalize_harness_id(value: object) -> str:
    """Return a canonical public harness ID, rejecting silent fallbacks.

    A missing value selects the product default.  Any supplied value must match
    an exact catalog ID so stale clients cannot accidentally launch a different
    harness than the owner selected.
    """

    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_HARNESS_ID
    if not isinstance(value, str):
        raise ValueError("harness ID must be a string")
    candidate = value.strip().lower()
    if candidate not in HARNESS_CATALOG:
        raise ValueError(f"unsupported harness ID: {value!r}")
    return candidate


def public_harness_catalog() -> list[dict[str, object]]:
    """Return the ordered, JSON-safe harness selector catalog."""

    return [
        {
            "id": item.harness_id,
            "label": item.label,
            "description": item.description,
            "default": item.harness_id == DEFAULT_HARNESS_ID,
        }
        for item in HARNESS_CATALOG.values()
    ]
