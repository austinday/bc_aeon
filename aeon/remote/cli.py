"""Command-line setup and server entry point for Aeon Remote."""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import sqlite3
import sys

from .config import RemoteConfig
from .controller_lock import ControllerLock
from .security import AuthService, generate_totp_secret, totp_uri
from .store import RemoteStore


def _config(*, validate_server: bool) -> RemoteConfig:
    config = RemoteConfig.from_env(validate_server=validate_server)
    config.prepare_state()
    return config


def init_admin(args) -> int:
    config = _config(validate_server=False)
    store = RemoteStore(config.database_path)
    auth = AuthService(store, config)
    username = args.username.strip()
    if not username or len(username) > 128:
        print("Username must contain 1-128 characters.", file=sys.stderr)
        return 2
    minimum_length = 8 if getattr(args, "allow_short_password", False) else 14
    password = getpass.getpass(f"New password ({minimum_length}+ characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        print("Passwords do not match.", file=sys.stderr)
        return 2
    try:
        password_hash = auth.hash_password(password, minimum_length=minimum_length)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Drop the plaintext references before touching durable state. CPython cannot
    # guarantee in-memory string erasure, but the value is never logged, placed in
    # argv, or written to disk.
    password = ""
    confirmation = ""
    password_only = bool(getattr(args, "password_only", False))
    secret = "" if password_only else generate_totp_secret()
    try:
        store.put_user(username, password_hash, secret, replace=args.replace)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("\nAdministrator configured. Existing web sessions were revoked.")
    print("The password is stored only as an Argon2id hash.")
    if not password_only:
        print("Add this TOTP secret to your authenticator app:")
        print(secret)
        print("\nAuthenticator URI:")
        print(totp_uri(secret, username))
        print("\nKeep this output private; the secret is not written to project files.")
    return 0


def serve(args) -> int:
    try:
        address = ipaddress.ip_address(args.host)
    except ValueError:
        address = None
    if (address is None or not address.is_loopback) and not args.allow_network_bind:
        print(
            "Refusing a non-loopback bind. Put Aeon Remote behind an HTTPS reverse "
            "proxy, or pass --allow-network-bind deliberately.",
            file=sys.stderr,
        )
        return 2
    config = _config(validate_server=True)
    store = RemoteStore(config.database_path)
    if store.admin_count() == 0:
        print("No administrator exists. Run: aeon-remote init-admin", file=sys.stderr)
        return 2
    from .app import create_app

    try:
        import uvicorn
    except ImportError:
        print("Install remote dependencies with: pip install '.[remote]'", file=sys.stderr)
        return 2
    app = create_app(config, store=store)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
        server_header=False,
        date_header=False,
    )
    return 0


def status(args) -> int:
    del args
    config = RemoteConfig.from_env(validate_server=False)
    try:
        with ControllerLock.acquire_read_lease(config.state_dir) as read_lease:
            store = RemoteStore(
                config.database_path,
                read_only=True,
                controller_guard=read_lease.assert_current,
            )
            administrators = store.admin_count()
            instances = store.list_instances()
    except (OSError, RuntimeError, sqlite3.Error):
        print(
            "Aeon Remote state is unavailable, active, or has not been initialized.",
            file=sys.stderr,
        )
        return 2
    print(f"State: {config.state_dir}")
    print(f"Administrators: {administrators}")
    for item in instances:
        print(
            f"{item['id'][:8]}  {item['status']:<12}  {item['name']}  "
            f"{item['workspace']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aeon-remote")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-admin", help="Create or rotate the remote administrator")
    init.add_argument("--username", default="admin")
    init.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing user and revoke all of their sessions",
    )
    init.add_argument(
        "--allow-short-password",
        action="store_true",
        help=(
            "Permit an explicitly user-chosen 8-13 character password; the "
            "password is still read privately from the terminal, never argv"
        ),
    )
    init.add_argument(
        "--password-only",
        action="store_true",
        help="Disable TOTP only for a deployment with an explicit outer auth layer",
    )
    init.set_defaults(func=init_admin)

    run = sub.add_parser("serve", help="Run the localhost web service")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument(
        "--allow-network-bind",
        action="store_true",
        help="Allow binding beyond loopback (HTTPS is still required)",
    )
    run.add_argument(
        "--forwarded-allow-ips",
        default="127.0.0.1",
        help="Trusted reverse-proxy IPs for forwarded headers",
    )
    run.set_defaults(func=serve)

    show = sub.add_parser("status", help="Show the durable instance registry")
    show.set_defaults(func=status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
