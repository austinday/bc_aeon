# OpenCode harness operations

Aeon is still the Nexus product and agent kind. Its reasoning harness is now a
reviewed, per-instance choice:

- `opencode` is the default for new or migrated settings. It uses OpenCode for
  the agent loop and context management.
- `legacy-aeon` keeps the original Aeon decide/act/observe loop available for
  comparison and rollback.

The desired harness is selected in an Aeon tab's **Agent settings**. It takes
effect on the next verified start or restart. Nexus reports desired and applied
values separately; it does not label an already-running, pre-migration process
as either harness without a launch receipt.

## Install and inspect OpenCode

The only accepted native artifact is OpenCode `v1.18.27` for Linux x86-64. Its
release URL, archive size and SHA-256, extracted executable size and SHA-256,
and version probe are pinned in `aeon/harnesses/catalog.py`. The installer never
uses a binary from `PATH` and never replaces an existing invalid version
directory.

From the canonical Aeon source tree:

```bash
cd /home/aday/NexusAgentDashboard/bc_aeon
python3 -m aeon.harnesses.opencode_install status --json
python3 -m aeon.harnesses.opencode_install install --json
```

`status` is read-only. `install` downloads and verifies the pinned release, or
accepts a previously downloaded archive with `--archive /absolute/path`. The
default owner-private root is
`~/.local/share/aeon/opencode/versions/v1.18.27/`; set
`AEON_OPENCODE_HOME` or `--home` only to an absolute owner-private directory.
A ready status means that ownership, modes, link count, receipt, sizes,
digests, platform, and `--version` all match the pin. Do not repair an
`invalid` result in place; preserve it for diagnosis and resolve the exact
ownership or content problem before a fresh installation.

The upstream artifact and license are available from the
[OpenCode v1.18.27 release](https://github.com/anomalyco/opencode/releases/tag/v1.18.27)
and [source license](https://github.com/anomalyco/opencode/blob/v1.18.27/LICENSE).

## Runtime boundary

The OpenCode process never chooses a GPU, host, claim, concrete endpoint, or
model artifact. One Aeon supervisor owns one `BrokerServiceSession` for the
logical `aeon-qwen38-standard` service. Its authenticated loopback model gateway
calls `ensure_ready()` immediately before each model request, follows Fleet
endpoint promotion, and rewrites the public logical model to the wire model
proved by the current runtime profiles. Fleet remains the only allocator and
the coordinator remains authoritative; unavailable compute is durable waiting.

The supervisor generates an isolated owner-private OpenCode configuration for
each instance. Auto-update, sharing, project configuration, third-party plugins,
external skills, downloads, and OpenCode's generic shell, file-mutation, web,
and sub-agent tools are disabled. A local stdio MCP bridge exposes only the
reviewed Aeon tool allowlist. Calls still pass through Aeon's request authority,
tool resource policy, action validation, receipts, and result archive. In
particular:

- generic commands remain CPU-only and constrained by Aeon's command guard;
- GPU model, media, and batch work uses existing Fleet-aware tools and reviewed
  profiles;
- browser, credential, GitHub, provider, sub-agent, memory, and lifecycle tools
  retain their existing authorization checks;
- MCP stdout is reserved for protocol framing, while model and tool failures
  are reduced to bounded, sanitized errors.

Pinned OpenCode cannot import a top-level JSON-Schema `oneOf`; one such schema
would otherwise hide the entire MCP catalog. The bridge flattens only unions of
closed object branches for transport, preserving every property constraint and
their shared required fields. The concrete Aeon tool still enforces its exact
cross-field alternatives before execution, so mixed or incomplete edit forms
remain refused.

Do not add a direct model URL, coordinator call, device selector, alternate
allocator, or unrestricted OpenCode tool to this path. A new tool belongs in
Aeon's normal tool catalog with an explicit compute route and must be reviewed
before it is added to `EXPOSED_TOOLS` in `opencode_mcp.py`.

## Browser and multimodal behavior

The OpenCode harness uses Aeon's existing authenticated browser service, not
OpenCode web fetching. One random browser session and last-tab binding persist
across the MCP subprocesses used by a running agent. A named persistent Chrome
profile retains its cookies and site storage across agent runs. The service uses
headed real Chrome, Patchright, normal viewport and browser identity, and
human-like mouse, keyboard, scroll, drag, and form interactions. These measures
reduce avoidable bot-detection friction; they do **not** guarantee CAPTCHA
avoidance or authorize bypassing an access-control challenge. The agent must
report a challenge that requires owner interaction. Navigation remains limited
to validated public HTTP(S) destinations and all existing SSRF, upload, media,
credential, and per-profile isolation rules still apply.

OpenCode declares text-and-image input for the Fleet model. Startup performs
Aeon's model vision self-test before readiness, and the MCP bridge returns
bounded image content from browser and vision tool receipts to the next model
turn. `analyze_image`, browser screenshots, image editing/generation, and video
tools therefore retain their existing multimodal and Fleet routing. Setting
`AEON_SKIP_VISION_SELFTEST=1` leaves vision explicitly unverified and is intended
for narrow tests, not production readiness.

## Liveness and cleanup

An OpenCode turn uses 12 steps by default and accepts only 1 through 32. Its
wall deadline defaults to 900 seconds and is clamped to 30 through 1,800 seconds
by `AEON_OPENCODE_TURN_TIMEOUT_SECONDS`. Event lines, final text, stderr, model
requests, and model responses also have hard byte ceilings.

Owner stop, timeout, output overflow, normal exit, and signals terminate the
exact OpenCode process group, including its MCP child, before cleanup. The
supervisor then closes the browser session, loopback gateway, Fleet ticket, and
presence record independently. A failed Fleet release makes the process fail;
it is never reported as successful cleanup. A missing saved OpenCode session is
discarded and retried once as a fresh session. `/clear` deliberately discards
the saved context while preserving the workspace and browser profile.

Final model text is buffered, not published immediately. After each MCP call,
the bridge atomically signs the current legacy Worker's request contract,
durable-agent guard, research guard, and typed receipt ledger. The supervisor
publishes final text only when that state has the exact authority digest,
instance, workspace, authority class, nonce, and observed tool-call count, and
the legacy completion checks accept it. Missing, stale, cross-instance, or
tampered evidence fails closed. An ordinary answer may use no tools, but an
inspection or change claim may not; a write requires a later exact readback or
targeted validator. The HMAC key is a private binary file rather than process
environment text. Supervisor state must be disjoint from the workspace, and the
command sandbox strips all `AEON_OPENCODE_*` variables and masks the exact state
directory, including against `/proc`-discovered file paths.
Bounded sub-agents also strip the complete `AEON_OPENCODE_*` and benchmark GPU
capability namespaces; a delegated process must mint its own turn and benchmark
authority instead of inheriting the principal's receipt paths or tokens.

SIGTERM and SIGHUP make the MCP process cooperatively stop every registered
receipt-bound synchronous command and release its process-local service
sessions before exit. Durable submitted Fleet jobs are IDs, not registered
cleanup callbacks, and continue under their normal durable lifecycle. The model
gateway separately tracks handler threads, upstream connections, detached
response sockets, responses, and downstream sockets. Cancellation shuts down
the sockets and joins the handlers before the gateway can reopen, so an OpenCode
retry cannot overlap a hung upstream request.

Continuous mode pauses for owner input after three identical OpenCode failures
or five consecutive failures, preventing an unchanged autonomous loop. A
non-interactive failed turn exits nonzero.

## Extending harnesses or models

Treat a harness/model choice as executable server policy, not free-form browser
input.

1. Add a stable harness ID and immutable metadata to
   `aeon/harnesses/catalog.py`. Native artifacts need exact platform, URL, size,
   digest, install receipt, and version verification.
2. Add a fixed, shell-free module argv in `aeon/harnesses/launch.py`; never accept
   a browser-supplied executable or arbitrary flags.
3. Implement the same Fleet-only model acquisition, tool authorization,
   cancellation, process-tree cleanup, transcript, presence, and desired/applied
   launch receipt contracts.
4. Add the choice to the server-owned agent settings catalog and migration-safe
   store constraints. Existing applied state must remain truthful.
5. For a model, add a logical Fleet service/profile mapping and release evidence
   first. Then add the server-owned model choice and compatible combinations;
   never put a concrete endpoint, host, GPU, or claim in a catalog.
6. Add benchmark provenance and tests for the exact new combination before it
   can be selected. Unsupported combinations must be rejected, not silently
   substituted.

## Verification and deployment

The model-free and fake-backed gates do not acquire a live lease:

```bash
cd /home/aday/NexusAgentDashboard/bc_aeon
python3 -m pytest -q \
  aeon/tests/test_opencode_install.py \
  aeon/tests/test_opencode_harness.py \
  aeon/tests/test_opencode_lifecycle_hardening.py \
  aeon/tests/test_agent_preferences.py \
  aeon/tests/test_remote.py
python3 -m compileall -q aeon/harnesses aeon/benchmarks
```

When the pinned binary is installed, the protocol integration test runs that
real executable against a fake streaming model and the real MCP file tool; it
does not request GPU compute:

```bash
python3 -m pytest -q aeon/tests/test_opencode_binary_integration.py
```

Follow `fleet_compute/docs/INTEGRATED_OPERATIONS.md` before deployment. Because
the OpenCode bridge adds the `mcp` package dependency, refresh the editable Aeon
install. Harness launch or `aeon.remote` changes and the Nexus selector require
a reconciled restart of `nexus-backend.service`, followed by restarting only the
affected Aeon agents so their applied harness is verified. A browser refresh
loads static UI changes. Do not restart Fleet Compute for harness-only changes;
Fleet needs its own reconciled restart only if an enabled profile, adapter, or
entry-point contract changed.
