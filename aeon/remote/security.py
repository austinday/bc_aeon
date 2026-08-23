"""Authentication primitives for the Aeon remote console."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError


def generate_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _decode_base32(secret: str) -> bytes:
    value = "".join(secret.upper().split())
    value += "=" * ((8 - len(value) % 8) % 8)
    return base64.b32decode(value, casefold=True)


def totp_code(secret: str, at_time: float | None = None, period: int = 30) -> str:
    counter = int((time.time() if at_time is None else at_time) // period)
    digest = hmac.new(_decode_base32(secret), struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    number = (struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF) % 1_000_000
    return f"{number:06d}"


def verify_totp(secret: str, supplied: str, at_time: float | None = None) -> bool:
    supplied = "".join((supplied or "").split())
    if len(supplied) != 6 or not supplied.isdigit():
        return False
    now = time.time() if at_time is None else at_time
    return any(
        hmac.compare_digest(totp_code(secret, now + offset * 30), supplied)
        for offset in (-1, 0, 1)
    )


def totp_uri(secret: str, username: str, issuer: str = "Aeon Remote") -> str:
    label = quote(f"{issuer}:{username}")
    return (
        f"otpauth://totp/{label}?secret={quote(secret)}&issuer={quote(issuer)}"
        "&algorithm=SHA1&digits=6&period=30"
    )


def token_digest(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def user_agent_digest(user_agent: str) -> str:
    return hashlib.sha256((user_agent or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LoginResult:
    token: str
    csrf_token: str
    expires_at: float
    max_age: int | None
    username: str


class AuthenticationError(Exception):
    pass


class LoginRateLimited(AuthenticationError):
    pass


# Argon2 is intentionally expensive (64 MiB in production).  This process-wide
# cap protects both the local login API and the public OIDC authorize endpoint
# from concurrent memory amplification across AuthService instances.
_ARGON_VERIFY_SLOTS = threading.BoundedSemaphore(2)


def _login_rate_key(domain: str, *parts: str) -> str:
    material = "\0".join((domain, *parts)).encode("utf-8", errors="replace")
    return hashlib.sha256(material).hexdigest()


class AuthService:
    def __init__(self, store, config, password_hasher: PasswordHasher | None = None):
        self.store = store
        self.config = config
        self.password_hasher = password_hasher or PasswordHasher(
            time_cost=3, memory_cost=65536, parallelism=2
        )
        # Missing usernames still pay a real verify cost, reducing username probes.
        self._dummy_hash = self.password_hasher.hash(secrets.token_urlsafe(24))

    def hash_password(self, password: str, *, minimum_length: int = 14) -> str:
        if minimum_length < 8:
            raise ValueError("Minimum password length cannot be below 8 characters")
        if len(password) < minimum_length:
            raise ValueError(
                f"Password must contain at least {minimum_length} characters"
            )
        return self.password_hasher.hash(password)

    def authenticate_password(
        self,
        username: str,
        password: str,
        *,
        client_ip: str,
        user_agent: str,
        remember: bool,
    ) -> LoginResult:
        """Authenticate the password-only Nexus flow without weakening TOTP mode.

        A deployment that explicitly enables TOTP must continue through
        ``authenticate`` with a valid code. This method is the narrow adapter used
        by the OIDC authorize form and the password-only local API.
        """

        return self.authenticate(
            username,
            password,
            "",
            client_ip=client_ip,
            user_agent=user_agent,
            remember=remember,
        )

    def authenticate(
        self,
        username: str,
        password: str,
        otp: str,
        *,
        client_ip: str,
        user_agent: str,
        remember: bool,
    ) -> LoginResult:
        username = (username or "").strip()
        normalized_peer = client_ip or ""
        rate_keys = (
            _login_rate_key("account-peer", normalized_peer, username.casefold()),
            _login_rate_key("peer", normalized_peer),
        )
        if not _ARGON_VERIFY_SLOTS.acquire(blocking=False):
            raise LoginRateLimited("Authentication is temporarily busy; try again shortly")
        try:
            attempt_id = self.store.reserve_login_attempt(
                {rate_keys[0]: 5, rate_keys[1]: 12}
            )
            if attempt_id is None:
                raise LoginRateLimited(
                    "Too many login attempts; wait before trying again"
                )

            user = self.store.get_user(username)
            password_ok = False
            try:
                password_ok = self.password_hasher.verify(
                    user["password_hash"] if user else self._dummy_hash,
                    password or "",
                )
            except (VerifyMismatchError, InvalidHashError):
                password_ok = False

            otp_ok = bool(user) and (
                not self.config.require_totp or verify_totp(user["totp_secret"], otp)
            )
            if not user or not password_ok or not otp_ok or not user["enabled"]:
                self.store.complete_login_attempt(attempt_id, succeeded=False)
                self.store.audit(
                    "login_failed",
                    actor=f"sha256:{_login_rate_key('audit-account', username.casefold())[:16]}",
                    client_ip=f"sha256:{_login_rate_key('audit-peer', normalized_peer)[:16]}",
                )
                raise AuthenticationError("Invalid credentials")

            self.store.complete_login_attempt(
                attempt_id,
                succeeded=True,
                clear_rate_keys=rate_keys,
            )
        finally:
            _ARGON_VERIFY_SLOTS.release()

        canonical_username = str(user["username"])
        lifetime = (
            self.config.remembered_session_days * 86400
            if remember
            else self.config.session_hours * 3600
        )
        raw_token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        expires = time.time() + lifetime
        self.store.create_web_session(
            user["id"], token_digest(raw_token), csrf, expires, user_agent_digest(user_agent)
        )
        self.store.audit(
            "login_succeeded", actor=canonical_username, client_ip=client_ip
        )
        return LoginResult(
            token=raw_token,
            csrf_token=csrf,
            expires_at=expires,
            max_age=lifetime if remember else None,
            username=canonical_username,
        )

    def session(self, raw_token: str | None):
        if not raw_token:
            return None
        return self.store.get_web_session(token_digest(raw_token))

    def logout(self, raw_token: str | None, *, client_ip: str = "") -> None:
        if not raw_token:
            return
        session = self.session(raw_token)
        self.store.revoke_web_session(token_digest(raw_token))
        self.store.audit(
            "logout",
            actor=session["username"] if session else "unknown",
            client_ip=client_ip,
        )
