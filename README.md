# Aeon

Aeon is an autonomous, self-modifying agent harness. A single control LLM drives a
plan→act→observe loop with collapsible tools, reusable skill protocols,
parallel sub-agents, persistent memory, and live context-pressure management.
Qwen3.8-27B is the only primary/control model and also handles image analysis
and browser screenshots. Tool-owned image, edit, video, and future specialist
models remain separate implementation details behind their tools.

## Quick start

```bash
python3 -m pip install --user -e .  # one-time install; creates the `aeon` command
cd /path/to/a/project
aeon                              # starts Qwen3.8 in this directory
aeon --start "Summarize this repo"
aeon -n --start "Do X"            # headless: exit when the objective completes
aeon doctor                       # read-only install/backend readiness check
```

`aeon` preserves the caller's working directory, stores project-local session
state beneath that workspace's `aeon_output/`, and loads applicable `AGENTS.md`
files from `~/AGENTS.md` down through the workspace hierarchy. More-specific
files are applied last. `aeon --help`, `aeon --version`, and `aeon doctor` use a
lightweight front door and never perform model/container cleanup.

The Qwen3.8 model is the locally built and validated artifact at
`$AEON_HOME/models/Qwen3.8-27B-ARA-abliterated-NVFP4-MTP`. Aeon does not
silently substitute a Hub model when it is missing or incomplete.

Native MTP depth is treated as a measured serving parameter, not an architecture
default. The historical direct-Docker `run_qwen38_mtp_sweep.sh` entrypoint is now
fail-closed: a future sweep must first be implemented as a reviewed benchmark mode
inside the same coordinator-verified, journaled Qwen lifecycle.
`benchmark_qwen38_mtp.py` remains an endpoint-side analysis client; it does not
launch GPU work. The recorded release sweep attempted 12 real-turn requests per K
across code recovery, browser screenshot grounding, safe system diagnosis, and
verified completion. A depth is eligible only when every response satisfies
the actual Aeon turn schema and repeatedly selects the exact intended tool and
arguments under the same deterministic profile used by the live control loop.
The release then chooses the highest
median decode throughput among eligible depths, preferring lower K within 1%,
and refuses to select any candidate below 100 tok/s.
The versioned result in
`aeon/core/data/qwen38_mtp_selection.json` is bound to the exact model
`BUILD_MANIFEST.json`, `SHA256SUMS`, and vLLM image ID. The production launcher
revalidates all three and refuses a guessed or stale draft depth.

The 2026-08-19 production-profile sweep selected **K=3** on a 48 GB RTX PRO
5000. K=0/1/2/3 each passed all 12/12 exact text, vision, browser, code,
system-diagnosis, and completion decisions at 59.12, 85.94, 99.18, and 103.51
median decode tok/s respectively. K=4 reached 101.48 tok/s but selected the
wrong action in 3/12 trials, so the correctness gate rejected it. The selected
K=3 profile also delivered 101.32 median end-to-end generation tok/s. The
runtime includes a source-hash-guarded backport of vLLM's upstream fix for
native MTP crossing Qwen's reasoning-to-structured-output boundary.
The historical 64k compact retest passed all 12/12 decisions at 103.93 median
decode and 101.58 median end-to-end tok/s. The promoted `.180` release now runs
the stronger measured 128k profile: dynamic per-token/head FP8 KV, MTP K=3,
8 scheduler sequences, and 8k chunked-prefill batches. Its release run passed
at 104.77 median serial decode tok/s, recalled an exact key from a measured
125,985-token prompt, and reached 529.76 aggregate decode tok/s at concurrency
8. The machine-readable receipt is in
`insights/qwen38_rtx5000_128k_20260822T1822/release_receipt.json`.

Common flags:

```bash
aeon --start "Summarize this repo"
aeon --start "Build X and test it" --max-iterations 40
aeon --help
```

| Flag | Purpose |
|------|---------|
| `--model NAME` | Skip the menu and use a specific model |
| `--menu` / `-i` | Open the optional account/configuration menu |
| `--dual` | Reserved for a future coordinator-safe dual-copy deployment; current releases use solo placement |
| `--start "..."` | Run an objective immediately, then drop into the REPL |
| `--max-iterations N` | Cap iterations per objective (forces a final report at the limit) |
| `--no-warmup` | Skip model warmup (faster startup) |
| `--debug` / `--debug-log PATH` | Verbose LLM/reasoning logging |

### Qwen compute placement

Aeon's default backend mode is `auto`. Once Fleet Compute is running and
advertises the enabled `aeon-qwen38-standard` service profile, Aeon requests an
expiring demand ticket over its owner-only Unix socket and consumes only the
sanitized ready endpoint. The broker owns placement, lease heartbeat, runtime
lifecycle, and idle scale-down. While that profile is not deployed, Aeon keeps
using its existing coordinator-managed compatibility lifecycle. A present but
unhealthy broker fails closed instead of falling back and risking two control
planes. Operators can require one path explicitly with
`AEON_COMPUTE_BACKEND=broker|coordinator`; ordinary starts should leave it at
`auto`.

The broker is not installed or started by Aeon. Its service/profile rollout is a
separate fleet-control operation and remains disabled until its adapter has the
required release evidence. The exact handoff and disabled profile draft are in
[docs/FLEET_COMPUTE_INTEGRATION.md](docs/FLEET_COMPUTE_INTEGRATION.md).

Aeon searches enabled release capabilities in deterministic fleet order. It
prefers `.177`, then tries each compatible worker/GPU through the central
coordinator. Raw free VRAM is never enough: a worker is eligible only after its
exact model, runtime, launcher, transport, resource profile and teardown path
have a machine-readable release receipt. Physical GPU 1 on `.177` remains
quarantined. `.179` has no `aday`-accessible release runtime and `.178`
currently lacks safe staging space. `.180` is enabled through the exact
`qwen38-compact-180-128k` remote-Docker capability on either coordinator-safe
physical GPU. Live renter ACL, lease, UUID, host-resource, and network checks
still decide whether it is usable for a particular start.

The orchestrating Aeon process remains on `.177`. A worker placement runs the
same UUID-pinned container behind a receipted loopback SSH tunnel. Heartbeats use
the exact remote container PID. If a renter preempts it, Aeon preserves the
durable job, proves the old process gone, releases only its exact claim, and
re-enters `.177`-first fleet admission. An unreachable or identity-ambiguous
worker is quarantined rather than force-released.

The `.177` profile uses a 48.7 GiB measured plan at 114688 tokens. The `.180`
profile uses the live-coordinator-admissible 41.25 GiB budget at 131072 tokens;
its largest sampled use during a 125,985-token prefill was 35,254 MiB. Each is a
measured aggregate peak plan and release accounting bound, not a generic PyTorch
or cgroup cap. Qwen uses the
fleet policy's exclusive-lease exception, pins every process to the coordinator
UUID, and retains at least 6 GiB for Vast. Because the claim is exclusive,
ComfyUI must use another coordinator-safe card or remain durably waiting.

Before reserve Aeon fully verifies the selected host's canonical model exact
checksummed file set, ownership and immutable permissions, plus the exact
preinstalled image ID.
The post-reserve transaction revalidates that inode/stat receipt, stages only its
small claim-owned source allowlist, and immediately before Docker create repeats
the exact coordinator claim, physical-node ACL, 96 GiB RAM, 96 GiB commit, 32 GiB
home and Docker-root disk, and 16 GiB shared-memory floors. CUDA selection uses
only the returned UUID; physical GPU 0 is diagnostic only for the ACL check.

The canonical Docker receipt binds the immutable container ID, random launch
nonce, exact image/argv/environment/labels, three read-only bind inodes, loopback
port, private 8 GiB `/dev/shm`, private 8 GiB compiler-cache tmpfs, finite PID
limit, non-root/read-only/cap-drop/no-new-privileges settings, and bounded local
Docker logs. Hugging Face and Transformers are offline and no global host cache
is mounted. Reuse and stop require that same full receipt; stop addresses only
the immutable ID. A durable releasing journal makes a crash between container
removal, coordinator release, and receipt cleanup idempotently recoverable.

`waiting_for_compute` means a live Aeon process is presently inside its
cancelable coordinator admission loop. Each reserve call and sleep is bounded;
the delay backs off from 15 seconds to a two-minute maximum, and no lease is held
between attempts. The loop remains foreground work in that Aeon/tmux process—no
daemon or background relaunch is created. Ctrl-C or **End agent** cancels the wait,
records `unavailable`, and leaves the durable Project Manager row resumable.
A process/service/machine interruption likewise leaves the row resumable, and
dead presence is never presented as an active queue.

## Aeon Remote

Aeon includes an optional authenticated, mobile-first web console for persistent
agent terminal tabs. It can create workspaces beneath configured roots, launch and
reattach to tmux-backed Aeon instances, stop or resume them, and display sanitized
host/coordinator resource usage. It does not expose a general browser shell.

Install with pip install '.[remote]', then follow the security and HTTPS setup in
[docs/AEON_REMOTE.md](docs/AEON_REMOTE.md). The included service and nginx files are
review templates; they are not installed or enabled automatically.

### Optional external expert

Aeon can keep Qwen3.8 as its only control model while exposing a tightly budgeted
`consult_external_expert` tool that is called automatically after two consecutive
failed local turns. The feature is
off by default. On interactive startup, choose **External expert account** from the
model picker to use an official Codex/ChatGPT, Claude Code, or Gemini CLI login; Aeon
stores only the provider choice, not the OAuth token. It sends no files
automatically and gives the hosted adviser no tool or execution authority. Before
anything is sent, local Qwen reviews the exact redacted prompt and blocks content
that may require an uncensored model, trigger hosted-model moderation/refusal, or
contain information the operator may not want disclosed to a large technology
company; uncertainty or review failure also blocks the call. A
separate API-key path is also supported. See
[docs/EXTERNAL_EXPERT.md](docs/EXTERNAL_EXPERT.md).

## How the loop works

Each iteration the harness assembles one prompt containing: core directives,
the visible tools, active skill protocol, persistent memories, a compressed
attempt log, runtime/GPU stats, the project tree, open files, the last result,
and a live sub-agent digest. The model replies with a JSON action plan
(`thought`, `intent`, `updated_plan`, `actions`, plus mandatory `skill_check` /
`memory_check` / `parallel_check` reflections). Multi-line content is passed via
`--- BEGIN BLOCK_N ---` blocks appended after the JSON, so code never has to be
escaped into JSON strings.

Built-in robustness:

- **Adaptive Qwen reasoning**: simple extraction, summarization, and browser
  actions use `low`; ordinary established turns use `medium`; planning, coding,
  ambiguous recovery, and retries use `xhigh`. Set
  `AEON_REASONING_EFFORT=low|medium|xhigh` to force a tier for benchmarking, or
  leave it unset for adaptive selection.
- **Separated thinking and constrained actions**: Qwen uses deterministic
  control sampling (`temperature=0`, `top_p=1`, `top_k=-1`, `min_p=0`) in a
  dedicated reasoning stream. Repeated production-schema trials showed that
  the former stochastic profile could change grounded tool decisions on
  identical evidence; greedy control is both more reliable and more favorable
  to MTP acceptance. The JSON grammar activates only after `</think>`, so hidden
  reasoning stays free-form while the final tool-action object cannot emit an
  unknown shape or invalid JSON.
- **Selective local search**: ambiguous browser challenges, failed/stalled turns,
  and difficult first decisions generate two or three independent local Qwen
  proposals. A separate grammar-constrained local verifier selects one using the
  current tests/command output, DOM/screenshots, or file contents/diffs; only that
  proposal executes. Routine work remains a single call. Set
  `AEON_LOCAL_SEARCH=off|adaptive|2|3|always` to override the policy.
- **Native thinking continuity**: real chat history is enabled by default and
  carries Qwen3.8's `reasoning_content` / `reasoning` fields with
  `preserve_thinking=true`. History is context-budgeted and persisted across
  restarts. Set `AEON_MESSAGE_HISTORY=0` only as a compatibility escape hatch.
- **Context pressure** is estimated every turn; the attempt log and memories are
  auto-compressed, and at >95% of the limit the largest/oldest open files are
  shed automatically instead of crashing.
- **Cross-run persistence**: memories, the plan, and the attempt log are written
  to `aeon_output/session_state.json` after every iteration. A fresh process
  restores durable memories (and, when the objective matches, the plan and log)
  instead of starting from amnesia — surviving a crash or clean exit, not just an
  in-process `restart_aeon`.
- **Verified edits**: `write_file` / `str_replace` return a compact unified diff
  of what actually changed (so a fuzzy/whitespace match landing in the wrong
  region is visible), plus a non-blocking syntax warning when a written
  `.py`/`.json`/`.yaml` file no longer parses.
- **Loop & stall detection**: identical command+output repeats, and the same
  intent repeated across turns, both trigger corrective nudges.
- **JSON recovery**: malformed output is repaired locally (trailing commas,
  Python literals) before falling back to a model-based repair.
- **Tool calls** auto-correct trivial name typos, suggest close matches on a
  miss, and report the expected signature on a bad-parameter call.
- **`run_command`** streams output, enforces its timeout even on silent hangs,
  and kills the whole process group (no orphaned GPU/CPU jobs).

## Tools and skills

Tools are grouped into collapsible categories to save context; top-level tools
(`run_command`, `open_file`, `write_file`, `str_replace`, `think`, `memorize`,
`spawn_sub_agent`, …) are always visible, and the agent calls
`expand_tool_category` / `collapse_tool_category` to reveal the rest (image,
video, browser, sub-agent coordination, …). Skills are pinned step-by-step
protocols the agent activates for matching objectives.

## Self-modification & self-improvement

Aeon can edit its own source, restart onto the new code, and measure whether the
change actually made it better — a guarded, reversible loop rather than a blind
overwrite.

- **Author skills by asking (full CRUD)**: `create_skill` (add), `read_skill`
  (inspect without activating), `create_skill(..., overwrite=true)` (modify), and
  `delete_skill` (remove). Every change is live the SAME session (skills are read
  from disk each turn — no restart for a skill that needs no new tool) and persists
  across restarts.
- **Add tools by asking**: drop a `BaseTool` subclass under `aeon/tools/` and call
  `restart_aeon`; the dynamic loader picks it up.
- **Zero-path self-modification**: `restart_aeon` and `verify_self_modification`
  auto-derive the agent's own source root from the installed package, so a
  self-modification "just works" from any workspace — the model never has to know
  or hand-supply the path to its own code, and verification always installs/tests
  the aeon source (not whatever project directory it happens to be running in).

- **Durable git checkpoints.** Every `restart_aeon` first tags the working tree as
  a recoverable checkpoint (`aeon/core/checkpoint.py`), capturing tracked *and*
  untracked files. Restores reconcile modifications, deletions, and additions so
  the `aeon/` package matches the checkpoint exactly, and never touch files
  outside it. Checkpoints persist as a diffable lineage (not a one-shot tarball).
- **Boot handshake (no bricking).** The pip/smoke/unit gates run the new code as a
  *subprocess*, but `execv` relaunches through the untested `--resume` path. A
  pending-boot marker (`aeon/core/bootguard.py`) names the checkpoint to roll back
  to; the relaunched process clears it only once it boots healthy, and any fresh
  start that still sees it auto-reverts. A resume-path crash can no longer leave
  broken code installed with no way back.
- **`revert_aeon`.** An agent-callable rollback to any checkpoint, for the case the
  gates can't catch: a change that passes tests but, once live, is a behavioural
  regression. Roll back, restart, continue.
- **Protected core.** A small constitution (`aeon/core/protected.py`) — the
  benchmark/scorer, the checkpoint & rollback engine, the boot handshake, the test
  gates — is refused by the edit tools unless a human sets
  `AEON_ALLOW_PROTECTED_EDIT=1`, so a self-modification can't quietly weaken the
  machinery that measures and reverts it.
- **A fitness signal.** `aeon/selfimprove/` turns one-off editing into a measurable
  loop: a scored **benchmark** of capability tasks run against a candidate in an
  isolated sandbox copy (`evaluate.py`), a **scorer** that accepts a change only if
  it holds or improves the score with no regressions (the ratchet), and a durable
  experiment **ledger**. `run_self_benchmark` exposes the score to the agent, so it
  can baseline before a change and confirm improvement (or trigger `revert_aeon`)
  after. Deterministic tasks need no model; model-driven behavioural tasks plug
  into the same registry.

## Web browsing (human-grade)

The `web_browser` tools drive **real Google Chrome** (not Chromium) via
**Patchright** (a patched Playwright that removes the CDP automation tells most
bot-detectors probe for), running **headed** under Xvfb (at a real-world
1920×1080) in a container (`aeon/services/browser/`) with a **persistent profile**
so logins/cookies survive. It uses no spoofed user-agent/viewport and no
detectable evasion shims; **every** action is physically human — the cursor
follows curved, time-sampled Bézier paths with an accelerate-then-settle velocity
and slight overshoot, clicks land on a randomized point within the element (not
dead center), keystrokes type at a real ~110 WPM cadence with word-boundary pauses
and occasional hesitations, scrolling rolls in wheel notches, drags
press-pause-move-release, checkboxes/switches are *clicked* (not toggled
programmatically), dropdowns are clicked open before selecting, fields are cleared
with select-all+delete, and even the pointer drifts a little while "reading."
Combined with the host's residential IP, the goal is to be indistinguishable from
a person.

The browser controller is not a public web service. Docker publishes it only on
`127.0.0.1:8030`, and every route—including health and API documentation—requires
a randomly generated bearer login stored at `~/.aeon/browser_api_token` with mode
`0600`. Aeon's browser tools supply that credential automatically. The launcher
refuses stale browser images that predate authentication, so it cannot silently
fall back to the old unauthenticated controller.

The human-v6 browser contract also provides semantic `browser_find` across
off-screen content, iframes, and open shadow roots; grounded visible page text on
every observation; explicit wait success/timeouts; privately staged workspace
uploads; and bounded HTTP/XHR/console failure events. A minimal window manager
gives headed Chrome a real 1920×1080 desktop instead of Xvfb's small default
window. It also supports screenshot-coordinate interaction for canvas/map/remote
desktop UIs, explicit native-dialog responses, and structured `browser_extract`
for forms, tables, links, and page text. Named persistent profiles can be selected
with `--browser-profile NAME` or `AEON_BROWSER_PROFILE=NAME`. The launcher requires
the human-v6 image marker, so an older authenticated
container cannot silently omit these capabilities.

Perception is **first-person**: the agent *is* the multimodal Qwen3.8 model, so
each browser action attaches the **actual rendered screenshot** to its next turn —
it looks at the page exactly as a human would and decides from the pixels, not
from a secondhand text caption. That view is paired with a **stable, indexed
element list** built from the accessibility tree — each interactable/meaningful
node stamped with a `data-aeon-id` and described by its role, accessible name,
value, and state (`expanded`/`collapsed`, `selected`, `checked`, `disabled`,
off-screen, scroll-group). So the model sees the page *and* has exact `[id]`s to
act on (no selector guessing). By default the screenshot carries small numbered
`[id]` marks (Set-of-Mark, for precise grounding); `visual="clean"` gives the pure
render, `visual="both"` gives both, and `include_vision=false` skips the image for
a faster element-list-only turn. Human-v6 additionally pairs the full 1920-pixel
frame with up to two lossless 2× PNG crops around the resolved intended control,
verification/CAPTCHA panel, small validation error, or dense table/diagram. The
service captures target geometry before acting, so a post-action re-render cannot
silently reuse the same numeric id and crop the wrong control.

Actions (`browser_interact`) cover the full human repertoire by id: click /
double / right-click, hover, type (real keystrokes, optional submit), press_key,
**scroll** (page or within a specific scroll container — e.g. an inbox list),
drag, press-and-hold, select_option, check/uncheck, upload_file, back/forward/
reload, wait_for, and `read_text` (clean readability extraction). Clicks can pass
`expected_text`, which is verified against the target before clicking to prevent
wrong-element clicks. `browser_read` re-observes without acting.

It also handles the things real sites throw at you: the element index descends
into **iframes and shadow DOM** (marked «in iframe»); native JS **dialogs** are
handled — but a `confirm`/`prompt` that looks **destructive** (delete/discard/
overwrite…) is *dismissed*, not auto-confirmed, and reported, so the agent never
silently deletes something; **downloads** are captured and copied into the
workspace `./downloads` so the agent can use them; **PDFs** (whose text isn't in
the DOM) are fetched and saved for parsing on `read_text`; button-triggered **file
pickers** work for uploads; **popups/OAuth windows** are captured as switchable
tabs; and a crashed browser is transparently relaunched.

More that keeps it human and capable:
- **Identity matches the network.** On launch the browser's timezone, locale, and
  `navigator.languages` are set from the **egress IP's geolocation** (with matching
  geo-coordinates), so an IP-vs-clock/language mismatch — a top bot signal — never
  shows. Override with `AEON_BROWSER_TZ` / `AEON_BROWSER_LOCALE`.
- **Proxy + leak protection.** Set `AEON_BROWSER_PROXY`
  (`http://user:pass@host:port` or `socks5://…`) to appear from anywhere; WebRTC
  is prevented from leaking the real IP around the proxy.
- **Idle motion + adaptive timing.** The cursor makes small "reading" drifts
  between actions (behavioral realism), and each action waits only until the DOM
  actually settles (fast on stable pages, patient on churning ones) instead of a
  fixed sleep.
- **See what changed.** `browser_interact(..., compare=true)` attaches the
  before *and* after screenshots so the model can diff its own action.
- **Per-agent isolation.** Each agent gets its OWN browser context (own cookie
  jar, storage, history, fingerprint): the principal uses the persistent `default`
  profile so logins survive across runs, while each sub-agent browses in an
  isolated context (`agent-<id>`) — so parallel agents are independent identities
  that never collide, and a sub-agent's context is torn down when it finishes.
- **Full navigation.** Multiple named tabs with switch/close; site-opened
  tabs/popups (target=_blank, window.open, OAuth) are captured, announced, and
  usable/closable; self-closing popups (OAuth) are detected and pruned; scrolling
  works both vertically AND horizontally, page-level or within a specific pane.
- **Challenges.** A `web/solving_challenges` skill guides the vision model through
  CAPTCHAs, image grids, sliders, and verification walls the human way.

## Self-modification

Aeon can edit its own source. The workflow is: change the code →
`verify_self_modification` (sandboxed sub-agent test) → `restart_aeon`. A restart
backs up the code, reinstalls, and **only relaunches if both the smoke test and
the unit tests pass**; otherwise it restores the backup and keeps the old code
running. All additions must be portable — Aeon runs from any project directory
and must not depend on its own source being in the workspace.

## Testing

```bash
./run_tests.sh                     # smoke test + unit tests (no GPU/model needed)
python3 -m aeon.smoke_test         # imports, tool discovery, syntax
python3 -m aeon.tests.test_core    # pure-logic unit tests
```

These are the same gates the restart path runs, so they are the fast way to
confirm a change is safe before going live.
