"""Pure, dependency-free helpers for the browser service.

Kept out of server.py (which imports Playwright/Patchright and can only run inside
the container) so this decision logic — proxy parsing, destructive-dialog
detection, timezone validation, locale selection — is unit-testable in the plain
harness environment. server.py imports these by bare name (`from browser_util
import ...`) because the container runs `uvicorn server:app` from /app where both
files sit side by side.
"""
import hmac
import os
import re
import stat
from types import MappingProxyType
from typing import Optional
from urllib.parse import urlparse


MIN_AUTH_TOKEN_BYTES = 32
BENCHMARK_SESSION_ORIGIN = "https://aeon-benchmark.invalid"
_BENCHMARK_SESSION_STORAGE_PREFIX = "aeon-benchmark-session-v1:"


# These documents are compiled into the authenticated browser service image.
# The benchmark endpoint accepts only an ID from this closed catalog; it never
# accepts HTML, script, a URL, or a filesystem path from its caller.
BENCHMARK_FIXTURES = MappingProxyType(
    {
        "observe-v1": """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Controlled observation fixture</title><style>body{font:24px sans-serif;padding:48px}#value{margin-top:30px;font-weight:bold}</style></head>
<body><h1>Observation Console</h1><p id="value">Preparing controlled value</p>
<script>setTimeout(()=>{document.querySelector('#value').textContent='ORBIT-5521';document.body.dataset.benchmarkState='ready'},80)</script></body></html>""",
        "form-v1": """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Controlled form fixture</title><style>body{font:22px sans-serif;padding:40px}label{display:block;margin:16px}input,select,button{font:inherit}</style></head>
<body><h1>Registration Exercise</h1><form id="form"><label>First name <input id="first" required></label>
<label>Email <input id="email" type="email" required></label><label>Team <select id="team" required><option value="">Choose</option><option value="research">Research</option></select></label>
<label><input id="agree" type="checkbox" required> Accept test terms</label><button type="submit">Submit registration</button></form><p id="result"></p>
<script>document.querySelector('#form').addEventListener('submit',e=>{e.preventDefault();if(!e.target.reportValidity())return;
if(document.querySelector('#first').value==='Ada'&&document.querySelector('#email').value==='ada@example.invalid'&&document.querySelector('#team').value==='research'&&document.querySelector('#agree').checked){document.body.dataset.benchmarkState='form-complete';document.querySelector('#result').textContent='Registration accepted'}else{document.querySelector('#result').textContent='Values do not match the exercise'}})</script></body></html>""",
        "session-v1": """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Controlled session fixture</title><style>body{font:24px sans-serif;padding:48px}button{font:inherit;margin:12px}</style></head>
<body><h1>Session Exercise</h1><button id="signin">Sign in to fixture</button><button id="continue" disabled>Continue session</button><p id="status">Signed out</p>
<script>const sid=new URL(location.href).searchParams.get('session');const key='aeon-benchmark-session-v1:'+sid;
const signed=localStorage.getItem(key)==='authenticated';document.querySelector('#continue').disabled=!signed;document.querySelector('#status').textContent=signed?'Signed in':'Signed out';
document.querySelector('#signin').onclick=()=>{localStorage.setItem(key,'authenticated');document.querySelector('#continue').disabled=false;document.querySelector('#status').textContent='Signed in'};
document.querySelector('#continue').onclick=()=>{if(localStorage.getItem(key)==='authenticated'){document.body.dataset.benchmarkState='session-complete';document.querySelector('#status').textContent='Session preserved'}}</script></body></html>""",
        "vision-v1": """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Controlled visual fixture</title><style>body{margin:0;background:#f4f4f4}.stage{height:900px;display:flex;align-items:center;justify-content:space-evenly}.shape{width:230px;height:230px;border-radius:50%;background:#d400d4;box-shadow:0 8px 22px #555}</style></head>
<body><main class="stage" aria-label="visual test"><div class="shape"></div><div class="shape"></div><div class="shape"></div></main></body></html>""",
    }
)

BENCHMARK_VERIFY_SCRIPTS = MappingProxyType(
    {
        "observe-v1": "document.body.dataset.benchmarkState === 'ready'",
        "form-v1": "document.body.dataset.benchmarkState === 'form-complete'",
        "session-v1": "document.body.dataset.benchmarkState === 'session-complete'",
        "vision-v1": "document.querySelectorAll('.shape').length === 3",
    }
)


def benchmark_fixture_definition(fixture_id: str):
    """Return one immutable HTML/verification pair or ``None``."""

    html = BENCHMARK_FIXTURES.get(fixture_id)
    verification = BENCHMARK_VERIFY_SCRIPTS.get(fixture_id)
    if html is None or verification is None:
        return None
    return html, verification


def validate_benchmark_fixture_request(
    session_id: str,
    tab_id: str,
    profile: str,
    fixture_id: str,
    operation: str,
):
    """Validate the complete non-model benchmark endpoint request."""

    if (
        re.fullmatch(r"oc-[0-9a-f]{32}", str(session_id or "")) is None
        or tab_id != "benchmark"
        or re.fullmatch(r"benchmark-[0-9a-f]{12}", str(profile or "")) is None
        or operation not in {"seed", "reopen", "verify", "cleanup"}
        or (operation in {"reopen", "cleanup"} and fixture_id != "session-v1")
    ):
        return None
    return benchmark_fixture_definition(fixture_id)


def benchmark_fixture_page_url(fixture_id: str, session_id: str) -> str:
    """Return the one route-intercepted origin used for session persistence."""

    if (
        fixture_id != "session-v1"
        or re.fullmatch(r"oc-[0-9a-f]{32}", str(session_id or "")) is None
    ):
        return "about:blank"
    return f"{BENCHMARK_SESSION_ORIGIN}/session-v1?session={session_id}"


async def seed_benchmark_fixture_page(
    page,
    fixture_id: str,
    *,
    session_id: str | None = None,
    reset_session: bool = False,
) -> bool:
    """Materialize only a catalog document on a supplied browser page."""

    definition = benchmark_fixture_definition(fixture_id)
    if definition is None:
        return False
    html, _verification = definition
    if fixture_id == "session-v1":
        url = benchmark_fixture_page_url(fixture_id, str(session_id or ""))
        if url == "about:blank":
            return False

        async def fulfill(route):
            await route.fulfill(
                status=200,
                content_type="text/html; charset=utf-8",
                body=html,
            )

        await page.route(url, fulfill)
        await page.goto(url, wait_until="domcontentloaded", timeout=10_000)
        if reset_session:
            await page.evaluate(
                "localStorage.removeItem('aeon-benchmark-session-v1:' + "
                "new URL(location.href).searchParams.get('session'))"
            )
            await page.reload(wait_until="domcontentloaded", timeout=10_000)
    else:
        await page.set_content(html, wait_until="domcontentloaded", timeout=10_000)
    await page.wait_for_timeout(120)
    return True


async def verify_benchmark_fixture_page(page, fixture_id: str) -> bool:
    """Evaluate only the catalog-owned success predicate for a fixture."""

    definition = benchmark_fixture_definition(fixture_id)
    if definition is None:
        return False
    _html, verification = definition
    return (await page.evaluate(verification)) is True


def read_auth_token(path: str) -> str:
    """Read a browser API bearer token from a private, regular file.

    Authentication must fail closed: accepting an absent, short, symlinked, or
    group/world-accessible token would turn the persistent logged-in browser
    profile back into an unauthenticated local service.
    """
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"browser authentication token is unavailable: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("browser authentication token must be a regular file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError("browser authentication token permissions must be 0600 or stricter")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            token = handle.read().strip()
    except OSError as exc:
        raise RuntimeError(f"browser authentication token could not be read: {path}") from exc
    if len(token.encode("utf-8")) < MIN_AUTH_TOKEN_BYTES:
        raise RuntimeError(
            f"browser authentication token must be at least {MIN_AUTH_TOKEN_BYTES} bytes"
        )
    return token


def bearer_is_authorized(authorization: str, expected_token: str) -> bool:
    """Constant-time validation of an ``Authorization: Bearer ...`` header."""
    scheme, separator, supplied = (authorization or "").partition(" ")
    if not separator or scheme.lower() != "bearer" or not supplied:
        return False
    return hmac.compare_digest(supplied.strip(), expected_token)

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
        if (
            not u.hostname
            or u.scheme.lower() not in {"http", "https", "socks5", "socks5h"}
            or u.path not in {"", "/"}
            or u.params
            or u.query
            or u.fragment
        ):
            return None
        if u.port is not None and not 1 <= u.port <= 65535:
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
