"""Pure, dependency-free helpers for the browser service.

Kept out of server.py (which imports Playwright/Patchright and can only run inside
the container) so this decision logic — proxy parsing, destructive-dialog
detection, timezone validation, locale selection — is unit-testable in the plain
harness environment. server.py imports these by bare name (`from browser_util
import ...`) because the container runs `uvicorn server:app` from /app where both
files sit side by side.
"""
from typing import Optional
from urllib.parse import urlparse

# Confirm/prompt dialogs whose message looks irreversible are DISMISSED rather
# than auto-confirmed, so the agent never silently triggers a destructive action.
DESTRUCTIVE_DIALOG_HINTS = (
    "delete", "remove", "discard", "erase", "overwrite", "unsaved",
    "permanently", "cannot be undone", "can't be undone", "are you sure",
)


def is_destructive_dialog(message: str) -> bool:
    """True if a confirm/prompt message reads as irreversible."""
    low = (message or "").lower()
    return any(k in low for k in DESTRUCTIVE_DIALOG_HINTS)


def parse_proxy(raw: str) -> Optional[dict]:
    """Playwright proxy dict from a URL like 'http://user:pass@host:port' or
    'socks5://host:port', or None if empty/unparseable. Never raises."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        u = urlparse(raw)
        if not u.hostname:
            return None
        scheme = u.scheme or "http"
        server = f"{scheme}://{u.hostname}" + (f":{u.port}" if u.port else "")
        prox = {"server": server}
        if u.username:
            prox["username"] = u.username
        if u.password:
            prox["password"] = u.password
        return prox
    except Exception:
        return None


def valid_timezone(tz: str) -> bool:
    """True if `tz` is a real IANA zone. A bad value from the IP lookup would make
    BOTH the Chrome and the Chromium-fallback launch throw, so callers validate
    before using it and fall back to a known-good default otherwise."""
    if not tz:
        return False
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(tz)
        return True
    except Exception:
        return False


def primary_locale(languages: str, default: Optional[str] = None) -> Optional[str]:
    """First locale from an ipapi 'languages' field like 'en-US,haw,fr' -> 'en-US'.
    Returns `default` when the field is empty/blank."""
    if not languages:
        return default
    first = languages.split(",")[0].strip()
    return first or default
