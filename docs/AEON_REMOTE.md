# Aeon Remote

Aeon Remote is a mobile-first, authenticated website for creating and managing
persistent terminal tabs that can enter Aeon, Codex, Claude, or Grok agent mode.
Browser disconnects do not stop a session: every tab runs in its own exact-name
tmux session. Shell creation accepts only an optional display name and validated
workspace; the server fixes the executable and assigns a collision-safe name when
the client omits one. Interactive shell input is available only after the existing
cookie, Origin, CSRF, and WebSocket checks.

An ordinary interactive `aeon` invocation outside tmux adopts itself into the
same managed registry and immediately attaches the invoking terminal. Existing
tmux/remote processes and `--non-interactive` runs do not nest another session;
`AEON_DISABLE_AUTO_TMUX=1` is the local emergency escape hatch.

## Security model

- Production startup requires an explicit HTTPS origin.
- Passwords are Argon2id hashes and TOTP is enabled by default.
- Login sessions are opaque random values stored only as SHA-256 digests in
  SQLite. Standalone Aeon Remote gives the browser an HttpOnly, Secure,
  SameSite=Strict, Host-only cookie.
- State-changing requests require a session-bound CSRF token and an exact allowed
  Origin.
- Terminal WebSockets require the same origin, cookie, protocol version, and CSRF
  value. An accepted socket revalidates that session every second and closes on
  logout, revocation, expiry, user disablement, or authentication-store failure.
- Login failures are rate-limited and lifecycle actions are recorded without
  terminal input/output, credentials, cookies, or TOTP values.
- Browser-created workspace paths must resolve beneath an explicit allowlist and
  symlink escapes are rejected. A locally adopted CLI may retain its exact
  already-authorized cwd outside that allowlist; the registry marks that launch
  origin explicitly, and only that recorded path is reusable on resume.
- The public service should bind only to loopback behind a dedicated HTTPS
  virtual host. Never port-forward Uvicorn directly.
- Force-stop and delete operations require the exact visible instance name.
- A managed shell's current pane directory is resolved and revalidated against the
  workspace allowlist before it can be used to start Aeon. The browser cannot
  provide that path to the start-here action.

The local state directory defaults to ~/.aeon/remote, mode 0700. Its SQLite
database and transcript files are mode 0600.

## Install

~~~bash
cd /home/aday/bc_aeon
python3 -m pip install '.[remote]'
~~~

Choose a dedicated HTTPS hostname and configure exact origins and workspace roots:

~~~bash
export AEON_REMOTE_ORIGINS=https://aeon.example.com
export AEON_REMOTE_HOSTS=aeon.example.com
export AEON_REMOTE_ALLOWED_ROOTS=/home/aday/aeon_workspaces:/home/aday/website_hosting
export AEON_REMOTE_STATE_DIR=/home/aday/.aeon/remote
~~~

Create the administrator locally. The command asks for a password without echoing
it and prints a new TOTP secret/URI once:

~~~bash
aeon-remote init-admin --username aday
~~~

Add that secret to an authenticator application, then keep the output private.
Running init-admin --replace rotates the password/TOTP secret and revokes every
existing web session for that user. A case-insensitive replacement also updates
the stored username to the exact requested casing without creating a duplicate.

`--password-only` is a scoped integration option for a deployment with an explicit
outer authentication layer; it is not the standalone default. Such a service must
also set `AEON_REMOTE_DISABLE_TOTP=1`. Password-only OIDC handoff uses
`SameSite=Lax` so the new session survives its cross-site callback, while HttpOnly,
Secure, CSRF, and exact-Origin protections remain enforced. Nexus uses this mode
to present one username/password screen with no email or TOTP prompt.

Passwords shorter than the normal 14-character provisioning minimum remain
rejected unless the operator deliberately supplies `--allow-short-password`. That
flag lowers only the interactive provisioning check to eight characters; the
password is still read without echo, never accepted in argv, and stored only as an
Argon2id hash.

## Local smoke test

Insecure HTTP is accepted only when explicitly enabled. Use this for loopback
testing, never for public access:

~~~bash
AEON_REMOTE_INSECURE_HTTP=1 \
AEON_REMOTE_ORIGINS=http://127.0.0.1:8765 \
AEON_REMOTE_HOSTS=127.0.0.1,localhost \
aeon-remote serve --host 127.0.0.1 --port 8765
~~~

Then open http://127.0.0.1:8765.

## Production topology

~~~text
phone browser -- HTTPS/WSS --> reverse proxy --> 127.0.0.1:8765
                                               Aeon Remote
                                                    |
                                               exact tmux tab
                                                    |
                                                   Aeon
~~~

The reverse proxy must:

- terminate a valid TLS certificate;
- redirect HTTP to HTTPS;
- preserve the original Host;
- proxy WebSocket upgrades;
- forward client IP headers only from the trusted local proxy;
- use a dedicated hostname, not a path under a less-trusted application.

See deploy/nginx-aeon-remote.conf.example. The service command itself must remain
bound to 127.0.0.1.

deploy/aeon-remote.service.example is a review template only. On the user's GPU
fleet, current operating policy forbids installing or enabling another persistent
daemon without an explicit policy exception. Do not install the unit merely because
the file exists.

## Terminal-first agent lifecycle

Browser-created work always begins as a managed terminal. The shell starts through
an empty environment with a private per-launch Bash rcfile and exact prompt/process
identity markers. The browser cannot supply an executable, argv, environment, or
agent objective. **Start Aeon here** and the provider start actions resolve the
pane's live current directory server-side, revalidate it against the allowlist, and
start a fixed agent command in that same tab. The tab ID, tmux session, instruction
versions, and local role remain stable. The legacy direct-agent creation endpoint
is intentionally disabled; its start-here alias performs the same in-place
activation.

No user-controlled shell command is constructed for activation. Aeon's own
coordinator-aware model launcher remains responsible for every GPU lease, UUID
selector, hard VRAM cap, and renter reserve.

- **Browser/web restart:** the tmux session continues and the browser reattaches to
  the same saved tab. Reconnects are bounded and tied to that tab's lifecycle
  generation so pre-transition input cannot reach a new foreground process.
- **End agent:** signals only the exact recorded managed foreground. Once the exact
  outer Bash prompt returns, the registry changes back to terminal mode. If process
  identity or tmux state is ambiguous, Nexus preserves an explicit error and
  requires exact-name force recovery; it never types `exit` into unknown work.
- **Stop terminal:** sends Ctrl-C and types `exit` only after proving the private
  prompt marker, Bash PID/session/TTY, and foreground process group.
- **Force stop:** requires the exact visible tab name, removes only the exact tmux
  session, and records stopped only after absence is proven.
- **Host reboot:** tmux does not survive. The database keeps the durable tab and
  marks a previously running session interrupted. **Reopen terminal** creates its
  fixed shell again; starting an agent remains a separate explicit action.
- **Delete:** removes only a proven-stopped registry row and preserves the workspace
  and Aeon recovery state. The pinned Project Manager/Home row cannot be deleted.

Managed terminal-first tabs do not enable a pipe-pane transcript in terminal or
agent mode. Older direct Aeon rows retain their bounded rotated transcript for
backward-compatible resume; transcript content is never included in Nexus status
or audit payloads.

Nexus extends this lifecycle with fixed native Codex, Claude Code, and Grok tabs.
Those provider processes start through an empty environment plus a small non-secret
runtime allowlist, retain each CLI's own approval flow and credential store, and do
not enable the Aeon transcript pipe. Provider login terminals likewise expose the
one-time flow only to the authenticated live WebSocket and never to audit/status
APIs.

Managed agent instruction profiles are locally known overlays, not vendor-hidden
system prompts. Aeon receives a private mode-600 JSON snapshot and reloads it when
building each system context. Codex receives a private named profile file and
Claude receives its documented private append-file argument; only file paths or
opaque profile names enter argv. Grok's private Nexus overlay is saved but not
injected because the reviewed CLI has no file-backed append contract. Applicable
workspace `AGENTS.md` continues to be loaded by Grok itself.

## Resource reporting

Host CPU, RAM, and disk metrics come from psutil. GPU information comes only from
the cooperative coordinator on DAY2RTX6000PRO; Aeon Remote never uses
nvidia-smi, never enumerates renter containers, and removes claim IDs, owners,
commands, PIDs, and GPU UUIDs from the browser response.

The resource panel is informational. Starting an agent never bypasses Aeon's own
coordinator reservation path. Agent activity exposes a bounded compute state
(`idle`, `waiting_for_compute`, `allocated`, or `unavailable`) and profile name,
but never a claim ID, GPU UUID, owner, command, or coordinator message.
