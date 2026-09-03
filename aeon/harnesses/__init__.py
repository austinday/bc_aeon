"""Modular agent-harness identities and installation support.

Importing this package is deliberately side-effect free.  In particular, it
does not inspect the network, create an OpenCode home, or start a subprocess.
"""

from .catalog import (
    DEFAULT_HARNESS_ID,
    LEGACY_AEON_HARNESS_ID,
    OPENCODE_ARCHIVE_SHA256,
    OPENCODE_EXECUTABLE_SHA256,
    OPENCODE_HARNESS_ID,
    OPENCODE_VERSION,
    normalize_harness_id,
    public_harness_catalog,
)

__all__ = [
    "DEFAULT_HARNESS_ID",
    "LEGACY_AEON_HARNESS_ID",
    "OPENCODE_ARCHIVE_SHA256",
    "OPENCODE_EXECUTABLE_SHA256",
    "OPENCODE_HARNESS_ID",
    "OPENCODE_VERSION",
    "normalize_harness_id",
    "public_harness_catalog",
]
