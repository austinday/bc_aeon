from __future__ import annotations

import pytest

from aeon.core.fleet_hosts import (
    COORDINATOR_ID_TO_HOSTNAME,
    COORDINATOR_ID_TO_NETWORK_ADDRESS,
    canonicalize_host_display_text,
    host_display_name,
    network_address,
)


def test_legacy_coordinator_ids_resolve_only_to_current_primary_lan_routes() -> None:
    assert COORDINATOR_ID_TO_NETWORK_ADDRESS == {
        "192.168.0.177": "192.168.8.111",
        "192.168.0.178": "192.168.8.114",
        "192.168.0.179": "192.168.8.112",
        "192.168.0.180": "192.168.8.113",
    }
    assert COORDINATOR_ID_TO_HOSTNAME == {
        "192.168.0.177": "DAY2RTX6000PRO",
        "192.168.0.178": "DAY2XRTX5000",
        "192.168.0.179": "DAY2XRTX6000-2",
        "192.168.0.180": "DAY2XRTX5000PRO-2",
    }
    assert all(
        address.startswith("192.168.8.")
        for address in COORDINATOR_ID_TO_NETWORK_ADDRESS.values()
    )


def test_user_facing_host_display_never_exposes_persisted_identity() -> None:
    assert host_display_name("192.168.0.177") == (
        "DAY2RTX6000PRO (192.168.8.111)"
    )
    rendered = canonicalize_host_display_text(
        "192.168.0.178 release failed; .180 fallback unavailable"
    )
    assert rendered == (
        "DAY2XRTX5000 (192.168.8.114) release failed; "
        "DAY2XRTX5000PRO-2 (192.168.8.113) fallback unavailable"
    )
    assert "192.168.0." not in rendered
    assert ".178" not in rendered
    assert ".180" not in rendered


def test_unknown_or_already_routed_address_cannot_bypass_identity_mapping() -> None:
    with pytest.raises(ValueError, match="outside the fixed Fleet routing table"):
        network_address("192.168.8.112")
    with pytest.raises(ValueError, match="outside the fixed Fleet routing table"):
        network_address("192.168.0.181")
    with pytest.raises(ValueError, match="outside the fixed Fleet routing table"):
        host_display_name("192.168.8.112")


def test_remote_adapter_ssh_argv_uses_current_route_not_legacy_identity() -> None:
    from aeon.core.qwen_dflash_training_adapter import HOST, NETWORK_HOST, _ssh_base

    assert HOST == "192.168.0.179"
    assert NETWORK_HOST == "192.168.8.112"
    command = _ssh_base()
    assert "aday@192.168.8.112" in command
    assert "aday@192.168.0.179" not in command
