"""Fixed Fleet identity-to-network routing boundary.

Fleet and coordinator persistence still use the original ``192.168.0.x`` host
keys. They are opaque compatibility identities, not routable addresses. All
network transports must resolve them through :func:`network_address` so an old
receipt or lease can never direct SSH to a retired subnet.
"""

from __future__ import annotations

import re
from typing import Final


COORDINATOR_ID_TO_NETWORK_ADDRESS: Final[dict[str, str]] = {
    "192.168.0.177": "192.168.8.111",
    "192.168.0.178": "192.168.8.114",
    "192.168.0.179": "192.168.8.112",
    "192.168.0.180": "192.168.8.113",
}

COORDINATOR_ID_TO_HOSTNAME: Final[dict[str, str]] = {
    "192.168.0.177": "DAY2RTX6000PRO",
    "192.168.0.178": "DAY2XRTX5000",
    "192.168.0.179": "DAY2XRTX6000-2",
    "192.168.0.180": "DAY2XRTX5000PRO-2",
}


def network_address(coordinator_id: str) -> str:
    """Return the exact live LAN route for a persisted Fleet host key."""

    try:
        return COORDINATOR_ID_TO_NETWORK_ADDRESS[coordinator_id]
    except KeyError as exc:
        raise ValueError("host is outside the fixed Fleet routing table") from exc


def host_display_name(coordinator_id: str) -> str:
    """Return a canonical user-facing hostname and current LAN address.

    Coordinator IDs remain valid persistence keys, but must never escape into a
    user-facing status surface as though they were current network routes.
    """

    address = network_address(coordinator_id)
    try:
        hostname = COORDINATOR_ID_TO_HOSTNAME[coordinator_id]
    except KeyError as exc:
        raise ValueError("host is outside the fixed Fleet routing table") from exc
    return f"{hostname} ({address})"


def canonicalize_host_display_text(detail: str) -> str:
    """Replace known internal host references in bounded diagnostic text."""

    rendered = str(detail)
    for coordinator_id in COORDINATOR_ID_TO_NETWORK_ADDRESS:
        rendered = rendered.replace(coordinator_id, host_display_name(coordinator_id))
    for coordinator_id in COORDINATOR_ID_TO_NETWORK_ADDRESS:
        shorthand = "." + coordinator_id.rsplit(".", maxsplit=1)[1]
        rendered = re.sub(
            rf"(?<![0-9]){re.escape(shorthand)}(?![0-9])",
            host_display_name(coordinator_id),
            rendered,
        )
    return rendered
