"""Operator-interruption regressions for CLI compatibility paths."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aeon import main


def test_service_wait_does_not_swallow_keyboard_interrupt() -> None:
    with patch.object(main.requests, "get", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            main.wait_for_service("test service", 12345, timeout=60)


def test_legacy_model_probe_does_not_swallow_keyboard_interrupt() -> None:
    with patch.object(main.requests, "get", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            main.get_ollama_models()


def test_retired_container_probe_does_not_swallow_system_exit() -> None:
    with patch.object(main.subprocess, "check_output", side_effect=SystemExit(4)):
        with pytest.raises(SystemExit, match="4"):
            main.is_container_running("retired")
