"""Constrain Fleet benchmark children to one exact loopback HTTP origin.

Python imports ``sitecustomize`` before the benchmark module.  The speed-lab
worker supplies the reviewed local port and puts this directory first on
``PYTHONPATH``.  This preserves hash-bound benchmark scripts while ensuring
their Requests calls cannot inherit host proxy settings or follow redirects.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import requests


def _port() -> int:
    raw = os.environ.get("AEON_LOCAL_HTTP_PORT", "")
    if not raw.isascii() or not raw.isdecimal():
        raise RuntimeError("AEON_LOCAL_HTTP_PORT is not an exact decimal port")
    value = int(raw)
    if not 1024 <= value <= 65535 or str(value) != raw:
        raise RuntimeError("AEON_LOCAL_HTTP_PORT is outside the reviewed range")
    return value


_LOCAL_PORT = _port()
_ORIGINAL_REQUEST = requests.Session.request


def _local_only_request(self, method, url, *args, **kwargs):
    if not isinstance(url, str) or len(url) > 4096:
        raise RuntimeError("Fleet benchmark HTTP URL is malformed")
    parsed = urlsplit(url)
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise RuntimeError("Fleet benchmark HTTP port is malformed") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed_port != _LOCAL_PORT
        or parsed.netloc != f"127.0.0.1:{_LOCAL_PORT}"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise RuntimeError("Fleet benchmark HTTP escaped its exact loopback origin")

    # Requests otherwise merges HTTP(S)_PROXY/ALL_PROXY into every call.  The
    # temporary trust_env change is safe here because benchmark helpers create
    # a fresh Session per module-level request, and explicit empty proxies give
    # the adapter the same property if Requests' merge behavior changes.
    previous_trust = self.trust_env
    self.trust_env = False
    kwargs["allow_redirects"] = False
    kwargs["proxies"] = {"http": "", "https": ""}
    try:
        return _ORIGINAL_REQUEST(self, method, url, *args, **kwargs)
    finally:
        self.trust_env = previous_trust


requests.Session.request = _local_only_request
