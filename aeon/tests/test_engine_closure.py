from __future__ import annotations

import os
from pathlib import Path

import pytest

from aeon.core.engine_closure import (
    EngineClosureError,
    load_engine_closure_receipt,
    measure_engine_closure,
    verify_engine_closure,
)


def _fixture(root: Path) -> dict[str, str]:
    (root / "bin").mkdir(mode=0o700)
    (root / "lib").mkdir(mode=0o700)
    (root / "lib/python3.12").mkdir(mode=0o700)
    (root / "lib/python3.12/site-packages").mkdir(mode=0o700)
    (root / "lib/python3.12/site-packages/package").mkdir(mode=0o700)
    (root / "bin/python3.12").write_bytes(b"exact-python\n")
    (root / "bin/python3.12").chmod(0o600)
    (root / "lib/python3.12/site-packages/package/__init__.py").write_bytes(
        b"VERSION = 1\n"
    )
    (root / "lib/python3.12/site-packages/package/__init__.py").chmod(0o600)
    (root / "bin/python").symlink_to("python3.12")
    return {"bin/python": "python3.12"}


def _receipt(measured: dict, links: dict[str, str]) -> dict:
    return {
        "archive_sha256": "a" * 64,
        "root": "venv",
        "python_executable_sha256": "b" * 64,
        "python_executable_bytes": 1,
        "python_version": "3.12-test",
        "python_cache_tag": "cpython-312",
        "python_soabi": "cpython-312-test",
        **measured,
        "allowed_symlinks": links,
    }


def test_complete_engine_closure_binds_every_path_and_file(tmp_path: Path) -> None:
    root = tmp_path / "venv"
    root.mkdir(mode=0o700)
    links = _fixture(root)
    measured = measure_engine_closure(root, links)
    receipt = _receipt(measured, links)

    assert measured["files"] == 2
    assert measured["symlinks"] == 1
    assert verify_engine_closure(root, receipt) == measured

    (root / "lib/python3.12/site-packages/package/__init__.py").write_bytes(
        b"VERSION = 2\n"
    )
    with pytest.raises(EngineClosureError, match="identity changed"):
        verify_engine_closure(root, receipt)


def test_engine_closure_rejects_path_set_and_inode_drift(tmp_path: Path) -> None:
    root = tmp_path / "venv"
    root.mkdir(mode=0o700)
    links = _fixture(root)
    measured = measure_engine_closure(root, links)
    receipt = _receipt(measured, links)

    (root / "unexpected.pyc").write_bytes(b"derived drift")
    (root / "unexpected.pyc").chmod(0o600)
    with pytest.raises(EngineClosureError, match="bound exceeded"):
        verify_engine_closure(root, receipt)
    (root / "unexpected.pyc").unlink()

    (root / "unreviewed-link").symlink_to("bin/python3.12")
    with pytest.raises(EngineClosureError, match="unreviewed or changed"):
        measure_engine_closure(root, links)
    (root / "unreviewed-link").unlink()

    os.mkfifo(root / "unsafe-fifo", mode=0o600)
    with pytest.raises(EngineClosureError, match="inode type is unsafe"):
        measure_engine_closure(root, links)


def test_reviewed_dev1141_receipt_is_the_complete_canonical_archive_closure() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "core/data/qwen38_v026_dev1141_engine_closure.json"
    )
    receipt = load_engine_closure_receipt(receipt_path)

    assert receipt == {
        "schema": "aeon-bare-engine-closure-v1",
        "archive_sha256": (
            "278fd5ac8447f73cb727f16d85717762c5f615fd5221bad7e2a350023d424baa"
        ),
        "root": "venv",
        "manifest_sha256": (
            "c93285eebb7a61bb988fd909f05f2bba575026bfbb6175408935c5a2c1d7b7c1"
        ),
        "entries": 81_903,
        "files": 73_103,
        "directories": 8_796,
        "symlinks": 4,
        "regular_bytes": 8_460_059_803,
        "python_executable_sha256": (
            "1e4461092175d186c4bcdb19ee2849a45d6e4d8d1f6a871e45f63418d52a4e35"
        ),
        "python_executable_bytes": 30_936_528,
        "python_version": (
            "3.12.13 (main, Jul 23 2026, 14:43:28) [Clang 22.1.3 ]"
        ),
        "python_cache_tag": "cpython-312",
        "python_soabi": "cpython-312-x86_64-linux-gnu",
        "allowed_symlinks": {
            "bin/python": "python3.12",
            "bin/python3": "python3.12",
            "bin/python3.12": "/usr/bin/python3.12",
            "lib64": "lib",
        },
    }
