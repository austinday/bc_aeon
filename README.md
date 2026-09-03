# Aeon

Aeon is Nexus's modular local-model agent. Pinned OpenCode is the default harness
for its agent loop, context management, and compaction; the original
plan→act→observe implementation remains selectable as `legacy-aeon`. Both use
the same reviewed Aeon tools, persistent browser and memory, multimodal inputs,
and Fleet-only logical model service. Concrete Qwen runtimes remain selected and
attested by Fleet rather than by either harness. Tool-owned image, edit, video,
and future specialist models remain separate implementation details behind their
tools. See
[OpenCode harness operations](docs/OPENCODE_HARNESS.md) and
[Aeon harness benchmarks](docs/BENCHMARKS.md).

## Quick start

```bash
python3 -m pip install --user -e .  # one-time install; creates the `aeon` command
python3 -m aeon.harnesses.opencode_install install --json  # verify/install pinned OpenCode
cd /path/to/a/project
aeon                              # starts Qwen3.8 in this directory
aeon --start "Summarize this repo"
aeon -n --start "Do X"            # headless: exit when the objective completes
aeon doctor                       # read-only install/backend readiness check
```

`aeon` preserves the caller's working directory, stores owner-private session
state outside source trees beneath `~/.aeon/state/` (or `AEON_STATE_DIR`), and
loads applicable `AGENTS.md` files from `~/AGENTS.md` down through the workspace
hierarchy. Old workspace `aeon_output/` checkpoints are migration reads only.
More-specific instruction files are applied last. `aeon --help`, `aeon
--version`, and `aeon doctor` use a lightweight front door and never perform
model/container cleanup. The enforced decide/act/observe contract is documented
in [docs/HARNESS_PROTOCOL.md](docs/HARNESS_PROTOCOL.md).

Aeon's logical/default display identity is
`Aeon Qwen3.8-Flash-Next 125B-A6B NVFP4+MTP`.
Fleet presence names the concrete routed profiles: Flash-Next lanes are reported
as Flash-Next, while the locally built artifact at
`$AEON_HOME/models/Qwen3.8-27B-ARA-abliterated-NVFP4-MTP` is reported explicitly
as the RTX 5000 fallback. The OpenAI-compatible wire token remains
`Qwen3.8-27B-ARA-NVFP4-MTP` only for zero-downtime compatibility with existing
READY fallback servers; it is not presented as the Flash artifact identity.
Aeon does not
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
revalidates those identities plus the immutable v5 suite and benchmark-script
provenance, and refuses a guessed or stale draft depth. The current v6 harness
was run separately as a K=3 release regression: its report SHA-256 is
`62f98e6a056fd0355dc1ce3d5d35c7bdd8729768c656ce32d91933f8764abc5c`, its
suite SHA-256 is
`b4148783023ad5bf95c174c5af2a6b0c2059d52183f33811cfaad91b98e22e5e`, and its
benchmark-script SHA-256 is
`a38cba76d5ffe73e9200b748311aaaa2f14593f0758ebf99f9191296672e0a1a`. That
15/15 K=3 regression confirms the selected release depth under the newer turn
schema; it is not, and does not claim to be, a replacement K=0..4 sweep.

The 2026-08-19 production-profile sweep selected **K=3** on a 48 GB RTX PRO
5000. K=0/1/2/3 each passed all 12/12 exact text, vision, browser, code,
system-diagnosis, and completion decisions at 59.12, 85.94, 99.18, and 103.51
median decode tok/s respectively. K=4 reached 101.48 tok/s but selected the
wrong action in 3/12 trials, so the correctness gate rejected it. The selected
K=3 profile also delivered 101.32 median end-to-end generation tok/s. The
runtime includes a source-hash-guarded backport of vLLM's upstream fix for
native MTP crossing Qwen's reasoning-to-structured-output boundary.
The historical 64k compact retest passed all 12/12 decisions at 103.93 median
decode and 101.58 median end-to-end tok/s. The promoted `DAY2XRTX5000PRO-2` release now runs
the stronger measured 128k profile: dynamic per-token/head FP8 KV, MTP K=3,
8 scheduler sequences, and 8k chunked-prefill batches. Its release run passed
at 104.77 median serial decode tok/s, recalled an exact key from a measured
125,985-token prompt, and reached 529.76 aggregate decode tok/s at concurrency
8. The sanitized machine-readable release receipt shipped with the package is
`aeon/core/data/qwen38_rtx5000_128k_release_receipt.json`; raw diagnostic reports
remain owner-local and are intentionally excluded from source publication.

Common flags:

```bash
aeon --start "Summarize this repo"
aeon --start "Build X and test it" --max-iterations 12
aeon --harness legacy-aeon --start "Run with the original Aeon loop"
aeon --help
```

| Flag | Purpose |
|------|---------|
| `--harness opencode\|legacy-aeon` | Select the agent loop; pinned OpenCode is the default |
| `--model NAME` | Skip the menu and use a specific model |
| `--menu` / `-i` | Open the optional account/configuration menu (`legacy-aeon` only) |
| `--dual` | Reserved for a future coordinator-safe dual-copy deployment (`legacy-aeon` only) |
| `--start "..."` | Run an objective immediately, then drop into the REPL |
| `--max-iterations N` | Cap turns per objective (OpenCode: 1–32; legacy: 1–10,000) |
| `--no-warmup` | Skip model warmup (`legacy-aeon` only) |
| `--debug` / `--debug-log PATH` | Verbose LLM/reasoning logging (`legacy-aeon` only) |

Interactive Aeon agents support the standalone `/clear` command at the normal
prompt, while work is running, and when the agent is waiting for an answer. It
clears that agent's conversation history, objective, plan, open-file context,
persistent memories, attempt history, and resumable session checkpoint. Runtime
and workspace system instructions, the Nexus persistent identity, model/settings,
tools, browser login profile, project files, and completed outputs are retained.

### Qwen compute placement

Aeon's production backend is Fleet Compute (`broker`; `auto` is a broker-only
alias). Once Fleet Compute is running and
advertises the enabled `aeon-qwen38-standard` service profile, Aeon requests an
expiring demand ticket over its owner-only Unix socket and consumes only the
sanitized ready endpoint. The broker owns placement, lease heartbeat, runtime
lifecycle, and idle scale-down. An absent or unhealthy broker fails closed; Aeon
never falls back to a second application allocator. Ordinary starts use
`AEON_COMPUTE_BACKEND=broker`.

The broker is installed as the single authorized `DAY2RTX6000PRO` owner service. Its
enabled, hash-bound Qwen and ComfyUI profiles have reviewed adapters. The exact
handoff is in
[docs/FLEET_COMPUTE_INTEGRATION.md](docs/FLEET_COMPUTE_INTEGRATION.md).

Nexus continuously maintains its pinned primary Aeon independently of any open
browser. If a reboot removes the managed tmux server, exact pane absence lets the
controller recreate that primary shell and reactivate the agent; ambiguous live
foreground state still fails closed. Fleet separately maintains the authenticated
replica target, so ready capacity is restored even when no agent ticket is active.

### Local video generation

`generate_video` acquires only the broker-managed `aeon-video-comfyui` service.
Fleet first tries the exact local `DAY2RTX6000PRO` runtime, then can stage the same
hash-bound image and models to coordinator-approved 48 GB cards on `DAY2XRTX5000` or
`DAY2XRTX5000PRO-2`; it never waits behind Aeon's always-on Qwen runtime on the only local
card when a safe worker is available. Its
default automatic route uses the hash-reviewed MiniMax H3 audiovisual stack:
10Eros-Max beta2 NVFP4, an uncensored Qwen3-VL-32B NVFP4 text encoder, and the
matching video/audio VAEs. H3 produces 24 fps H.264 MP4 with synchronized stereo
audio, first/last-frame conditioning, and timed multi-shot direction for clips up
to about 15 seconds. LTX-2.3 10Eros 1.5 Q8 remains the automatic specialist for
explicit motion, arbitrary interior keyframes, continuation, and editing.

The `generate_video` workflow compiles ordinary requests into H3's official three-part
audiovisual IR, verifies output media, and assembles longer work through the same
tool. A successful final video is copied into the owning agent's private transcript
attachment store; authenticated Nexus renders it as an inline player in that exact
agent tab. No model path, Comfy endpoint, Fleet ticket, claim, or GPU identity is
exposed to the browser.

### Source propagation and upgrades

Nexus and Fleet Compute import this canonical source tree directly. Ordinary
Python edits are visible to newly started processes without copying files, but a
running agent keeps modules already imported in memory. Restart the affected agent
through Nexus. Restart `nexus-backend.service` as well when changing `aeon.remote`,
provider launch/instruction assembly, or other Aeon modules imported by Nexus.
Restart `fleet-compute.service` at a reconciled maintenance point when changing
Qwen/ComfyUI adapters, their entry-point registration, or enabled Fleet profiles.
Dependency, console-script, or entry-point metadata changes require refreshing the
editable install with `python3 -m pip install --user -e .`; ordinary `.py` edits do
not. The canonical cross-project procedure is
`/home/aday/NexusAgentDashboard/fleet_compute/docs/INTEGRATED_OPERATIONS.md`.

Fleet Compute selects enabled service-profile variants in deterministic order and
prefers the reviewed `DAY2RTX6000PRO` local lane before the generic compact-worker lane.
Within that worker lane, placement considers `DAY2XRTX5000` before `DAY2XRTX5000PRO-2`; the Qwen
adapter then verifies the matching host-specific release capability against the
coordinator lease. Raw free VRAM is never enough: a worker is eligible only after
its exact model, runtime, launcher, transport, resource profile and teardown path
have a machine-readable release receipt. Physical GPU 1 on `DAY2RTX6000PRO` remains
quarantined. The enabled `aeon-qwen38-compact-workers` profile can use the exact
`qwen38-compact-180-128k` remote-Docker capability on either coordinator-safe
physical GPU of `DAY2XRTX5000PRO-2`. The `DAY2XRTX5000` placement is disabled after live startup
proved its leased UUID was no longer visible inside the NVIDIA container following
the hardware-topology change; it requires fresh qualification before re-enabling.
Those capabilities share one portable runtime contract but retain distinct,
host-bound qualification receipts; one host's receipt cannot authorize the
other. `DAY2XRTX6000-2` still has no enabled Qwen serving capability. Live renter ACL,
lease, UUID, host-resource, storage, and network checks still decide whether any
qualified placement is usable for a particular start.

The bounded `DAY2XRTX5000` qualification service is now disabled and retired after its
passed evidence and exact teardown. It is not a serving-pool lane. The legacy
host-specific `DAY2XRTX5000PRO-2` profile is also disabled; the generic compact-worker profile
is the sole production authorization for both qualified workers.

The orchestrating Aeon process remains on `DAY2RTX6000PRO`. A worker placement runs the
same UUID-pinned container behind a receipted loopback SSH tunnel. Heartbeats use
the exact remote container PID. If a renter preempts it, Fleet and the reviewed
adapter preserve durable demand, prove the old process gone, release only its exact
claim when allowed, and re-enter `DAY2RTX6000PRO`-first admission. An unreachable or
identity-ambiguous worker is quarantined rather than force-released.

The `DAY2RTX6000PRO` profile uses a 48.7 GiB measured plan at 114688 tokens. The compact
worker profile uses the live-coordinator-admissible 41.25 GiB budget at 131072
tokens. The `DAY2XRTX5000` READY-state sample observed 35,376 MiB used during its release
gate, while the `DAY2XRTX5000PRO-2` release observed 35,254 MiB during a 125,985-token
prefill. These are sampled release observations within the enforced aggregate
plan, not claims about historical peak use or current free capacity. The release
budgets are the measured aggregate peak plan and accounting bounds, not a generic
PyTorch or cgroup cap. Qwen uses the
fleet policy's exclusive-lease exception, pins every process to the coordinator
UUID, and retains at least 6 GiB for Vast. Because the claim is exclusive,
ComfyUI must use another coordinator-safe card or remain durably waiting.

Before and after Fleet reserves, the Qwen adapter verifies the selected host's
canonical model checksummed file set, ownership and immutable permissions, plus the
exact preinstalled image ID. The post-reserve transaction revalidates that
inode/stat receipt, stages only its
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

`waiting_for_compute` means a live Aeon process has durable broker demand but no
safe ready endpoint. Fleet's coordinator attempts and backoff are bounded, and no
consumer-side allocation loop or lease exists. The Aeon/tmux process polls only its
opaque ticket and remains cancelable—no second daemon or background relaunch is
created. Ctrl-C or **End agent** releases/cancels that demand, records
`unavailable`, and leaves the durable Project Manager row resumable.
A process/service/machine interruption likewise leaves the row resumable, and
dead presence is never presented as an active queue.

The Nexus Main orchestrator alone can use `start_agent_instance` to register another
standalone, durable Aeon or connected subscription-agent session in an existing
allowed directory. Ordinary Aeon instances and bounded `spawn_sub_agent` workers
cannot use it; bounded delegation still reports back into its parent. Nexus renders
the created record as a linked, renameable agent tab beside the pinned primary chat
without disclosing its host address in the tab label. A new Aeon tab is persisted
idle with no process, objective, or compute demand; its first user message becomes
the exact one-time startup objective and is not duplicated through terminal input.
When the user explicitly requests ongoing autonomous work, the same bridge can set
a three-or-more-word continuous goal, a persistent personality, and private system
instructions before launch. In that case Nexus starts the tab with continuous mode
already enabled. Aeon re-reads its owner-private control file between turns and
starts another goal-driven work cycle after `final`, `ask_user`, or `wait`; a queued
real user message still wins. Turning the mode off requests a cooperative stop of
the current turn and waits for the worker's durable stop acknowledgement. If an
active turn does not acknowledge within the bounded interval, Nexus stops the exact
verified Aeon process and relaunches it idle with the existing context, while leaving
continuous mode disabled. Ending/stopping also disables the mode without deleting
its saved goal. A fresh restart instead clears the objective, plan/checklist,
visible chat, and durable conversational state before launching an idle process; it
never replays the old objective or implicitly resumes continuous work. An
interrupted durable Aeon exposes **Start continuous** as a combined
enable-and-resume action, preserving its saved context and goal. Continuous prompts explicitly grant no new
authority and never reinterpret an unanswered question as user approval.

Nexus collaborator portals run a separate clean Aeon sibling with the target's
model/workspace/project association but none of its transcript, memories,
instructions, credentials, or continuous state. The sibling receives a dedicated
minimal liaison prompt and only `send_collaborator_handoff`. Handoffs include a
server-captured verbatim external-user excerpt as well as the model summary, enter
the target through its normal pending chat path, and retry idempotently on resume.
The target treats that input, and any following synthetic continuous cycles, as
untrusted and dialogue-only. A fresh owner request quarantines that influenced
context before normal authority can resume.
`create_collaboration_portal` only pins an owner-approval proposal; it never
creates credentials or public access itself. See
[`docs/AEON_REMOTE.md`](docs/AEON_REMOTE.md#collaborator-siblings) for the isolation,
preemption, restart, and fake-backed test contracts.

Every running Nexus-managed Aeon also receives the top-level `set_job_role` tool.
It accepts only new Job Role text, or an explicit reset to the shared default; it
has no instance selector and cannot edit shared defaults or another tab. Nexus
binds the request to a random owner-only capability issued for that launch, saves
the result in the existing versioned instance layer, and reloads it on the next
model turn. Nested sub-agents have the tool removed and the parent capability
scrubbed from their environment.

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

Each request is classified as answer, inspection, plan, local change, external
action, or destructive action. The model receives the exact request as a real
`role=user` message plus a stable system/tool prefix. Only harness-owned policy
metadata is added as `role=system`; volatile project, memory, open-file, receipt,
CPU/memory, and job state is explicitly untrusted and arrives as a synthetic
assistant tool call with a matching `role=tool` receipt. It returns one
constrained turn kind: `tool_calls`, `final`, `ask_user`, or `wait`.

The model proposes; the harness authorizes and records. Tool calls are preserved
as typed assistant calls and results as matching `role=tool` receipts. The
execution layer rejects effects outside the request mode, yields on ask/wait,
preempts stale decisions when a newer user message arrives, and blocks success
claims that lack current evidence. See
[docs/HARNESS_PROTOCOL.md](docs/HARNESS_PROTOCOL.md).

Managed Aeon sessions reload the canonical core, execution, reminder, primary,
Fleet-safety, main-orchestrator, and default-job-role prompt files for each prompt
build. Applicable workspace instructions are loaded before the private Nexus layer,
so a versioned agent-only setting can intentionally override the matching shared
default without changing other agents.

Built-in robustness:

- **Prefix-cache-stable prompt layout**: the invariant core, execution, private,
  workspace, and tool-catalog text is serialized before expandable tool/category
  and skill state. Changing the active category therefore preserves the largest
  possible byte-identical prefix for vLLM; live state remains in the final typed
  harness receipt and instruction bodies are still reloaded on every turn.
- **Adaptive Qwen reasoning**: simple extraction, summarization, and browser
  actions use `low`; ordinary established turns use `medium`; planning, coding,
  ambiguous task recovery, and ordinary retries use `xhigh`. A generation-budget
  recovery is deliberately different: it disables hidden thinking and uses one
  low-effort generation within an 8,192-token aggregate and 180-second wall
  budget, so an overlong reasoning loop cannot reproduce another 32K attempt. Set
  `AEON_REASONING_EFFORT=low|medium|xhigh` to force a tier for benchmarking, or
  leave it unset for adaptive selection.
- **Separated thinking and constrained actions**: Qwen uses deterministic
  control sampling (`temperature=0`, `top_p=1`, `top_k=-1`, `min_p=0`) in a
  dedicated reasoning stream. Repeated production-schema trials showed that
  the former stochastic profile could change grounded tool decisions on
  identical evidence; greedy control is both more reliable and more favorable
  to MTP acceptance. The JSON grammar activates only after `</think>`, so hidden
  reasoning stays free-form while the final object is decode-time constrained to
  one valid turn kind and exact tool-specific parameter schema.
- **Selective local search**: ambiguous browser challenges, failed/stalled turns,
  and difficult first decisions generate two or three independent local Qwen
  proposals. A separate grammar-constrained local verifier selects one using the
  current tests/command output, DOM/screenshots, or file contents/diffs; only that
  proposal executes. Routine work remains a single call. Set
  `AEON_LOCAL_SEARCH=off|adaptive|2|3|always` to override the policy.
- **Bounded evidence continuity**: typed chat history is enabled by default, but
  hidden provider reasoning is transient rather than being replayed as factual
  evidence. Complete assistant/tool groups are retained as a bounded durable
  suffix, while older groups become a deterministic chained digest. Set
  `AEON_MESSAGE_HISTORY=0` only as a compatibility escape hatch; the separate
  `AEON_PRESERVE_REASONING_HISTORY=1` diagnostic switch is not a production
  default.
- **Safe Nexus progress**: a Nexus-managed primary process appends compact progress
  records containing a redacted one-sentence intent and allowlisted tool names.
  Hidden reasoning, parameters, commands, prompts, and raw tool output never enter
  the browser transcript.
- **Context pressure** is estimated every turn; the attempt log and memories are
  auto-compressed, and at >95% of the limit the largest/oldest open files are
  shed automatically instead of crashing.
- **Cross-run persistence**: memories, the plan, the bounded evidence suffix,
  and the attempt log are written under owner-private
  `~/.aeon/state/workspaces/.../sessions/...` state after every observation
  boundary. Checkpoints have a symmetric 8 MiB read/write ceiling and atomic
  owner-private replacement. A fresh process restores durable memories and a
  matching waiting request without dirtying the source workspace. Old
  `aeon_output/` state is read only as a migration fallback.
- **Evidence-bound edits**: opening a file returns its SHA-256. Existing-file
  writes require that exact receipt; stale, blind, symlink, ambiguous, and
  fuzzy-by-default edits fail closed. Python/JSON/YAML syntax is checked before
  atomic replacement, and validation reads must target the changed file (or use
  a broader test/health probe) before completion.
- **Typed outcomes**: tool status is normalized to `ok`, `failed`, `blocked`,
  `pending`, or `no_change`. A mutation is an observation boundary, and no
  mutation request can finish without a change plus later validation or a
  current receipt proving the requested state already existed.
- **Finite loop & stall control**: repeated failures, refusals, or unchanged
  proposals are bounded and barred when non-retryable. A request has a hard
  64-decision ceiling; one decision has a shared six-call, 65,536-token,
  1,800-second local generation backstop, including recovery/search work. Each
  primary generation may use up to 32,768 output tokens, and the worker reserves
  that space when fitting the prompt into the served context. If that whole
  decision attempt exhausts its allowance, the worker makes one fresh compact
  low-effort decision with hidden thinking and candidate search disabled, an
  8,192-token aggregate ceiling, and a 180-second wall ceiling. The prompt fitter
  reserves that smaller output window. A candidate that alone reaches its length
  cap no longer suppresses an independent candidate that still fits the shared
  budget. Continuous recovery removes only synthetic failed-cycle prompt/message
  pairs from model history. Three semantically identical failures, or six
  consecutive failed/blocked cycles with changing error text, open a circuit
  instead of retrying forever. The owner-visible transcript, continuous
  setting and goal, tool evidence, and successful chat remain intact. These are
  finite runaway/liveness guards, not paid-token or cloud quotas.
- **Schema and runtime validation**: structured decoding prevents unknown tool
  names and impossible final/tool combinations; the executor independently
  rejects missing, unknown, and wrongly typed parameters.
- **Capability and target binding**: consequential requests authorize exact
  capability families and exact typed targets. GitHub, agent creation,
  publication, service control, deletion, and other external/destructive actions
  cannot substitute for one another, and a confirmation can bind only the
  proposal visibly awaiting that confirmation.
- **Curated parallel reads**: up to four reviewed, stateless read tools may run
  concurrently. Shell, mutation, Fleet lifecycle, active-model, and other
  stateful operations remain serialized observation boundaries.
- **Workspace-confined file tools**: direct reads and writes traverse beneath the
  immutable launch workspace with descriptor-relative, no-symlink checks. Hidden
  credential/state roots and multiply-linked files are refused; writes use exact
  inode/version preconditions and atomic replacement.
- **Explicit compute routes**: every registered tool is classified as local CPU,
  active-model, Fleet service/child, dynamic command, Nexus lifecycle, or external
  provider. Undeclared tools fail registration; local-model tool calls revalidate
  the active Fleet ticket immediately before execution.
- **`run_command` / `run_command_async`** run ordinary host work through
  `fleet-low-priority` in a cryptographically unique gated user-systemd service.
  The shell starts only after exact unit/MainPID/cgroup/InvocationID and hardened
  property readback is durably receipted. Accelerator visibility and inherited
  Fleet lease authority are scrubbed. Seccomp denies `socket`/`socketpair`, so
  generic commands have no DNS, loopback, public Internet, or AF_UNIX access;
  reviewed browser/search/provider tools own network effects.
  Admission rejects direct GPU/coordinator/device/lease access, every container or
  runtime client (whose daemon would escape the service), privilege/scope/namespace
  launchers, remote execution, service/scheduler mutation, generic process signals,
  and recognized GPU/distributed launches before the requested command or
  background-job directory exists. Landlock independently denies non-standard
  devices, the coordinator, service/broker sockets, and credential stores. The
  Aeon source/guardrail tree is read-only; only an exact private temporary path is
  writable there and is removed at service collection. External workspaces with
  no protected overlap retain ordinary cwd writes. GPU work enters through a
  reviewed Fleet Compute service or batch profile.

For model-selected batch work, `fleet_batch_capabilities` is the executable
preflight. It lists only goal-eligible, source-reviewed recipes; AGENTS.md rules
do not create a launcher. `fleet_submit_batch_job` binds a listed recipe to its
fixed profile/project/payload and returns a durable owned job receipt, while
`fleet_batch_job_status` can read only jobs submitted by that exact managed
agent. Continuous mode backs off between pending checks rather than polling or
submitting duplicate demand. An empty catalog is a stable “no reviewed recipe”
answer, not permission to search for direct GPU access.

## Tools and skills

Tools are grouped into collapsible categories to save context; top-level tools
(`run_command`, `open_file`, `write_file`, `str_replace`, `memorize`,
`spawn_sub_agent`, …) are always visible when authorized, and the agent calls
`expand_tool_category` / `collapse_tool_category` to reveal the rest (image,
video, browser, sub-agent coordination, …). Communication/completion is handled
by the turn envelope rather than duplicate tools. The baked-in skills are
broad advisory playbooks. Each durable agent also has a private, bounded learned
skill overlay and searchable wiki; neither is treated as instruction authority.

GitHub work uses five distinct top-level capabilities rather than networking from
`run_command`: `github_repositories`, `github_status`, `github_commit`,
`github_push`, and `github_verify_remote`. Nexus binds each request to the managed
agent's exact workspace and one explicitly allowed GitHub credential. Status
returns paths and diff summaries without file contents; commit accepts only exact
relative paths and remains a local mutation; push is a separately authorized,
non-force external mutation followed by an exact remote-head comparison. The
credential stays behind Nexus's loopback gateway and never enters the agent,
command sandbox, tool receipt, or repository hook/configuration environment.
Remote operations fail closed unless provider metadata proves the exact credential
account personally owns a private repository. An unverified push is a typed failure
that requires exact remote verification; local ref drift during the provider
round-trip is likewise a typed failure after exact HEAD readback. An
all-current-changes backup requires
a final clean typed status rather than a successful partial commit or push.

Hugging Face publication likewise uses three top-level typed capabilities:
`huggingface_account`, `huggingface_publish_model`, and
`huggingface_verify_publication`. Nexus—not the model process—reads the one
explicitly assigned credential, authenticates its writable namespace, and binds
all local files to the managed agent's exact workspace. New repositories are
staged privately; a publish receipt is successful only after the remote file set,
commit, and requested visibility are read back. Ambiguous uploads must be verified
before retrying or claiming success.

## Self-modification & self-improvement

Aeon can edit its own source, but candidate imports and automatic re-exec remain
fail-closed unless the host can actively prove their stronger isolation boundary.
This keeps source authoring available without pretending an unsafe restart or
benchmark is verified.

- **Earn skills from experience**: `remember_skill_knowledge`,
  `search_skill_knowledge`, `read_skill_knowledge`, and
  `delete_skill_knowledge` maintain editable per-agent offline context. A
  first-try success can become a note, but `create_skill` requires exact
  low-uncertainty evidence of a failed approach followed by verified recovery.
  One recovery episode earns at most one shortcut; extra findings stay in the wiki.
  `read_skill`, revision-bound `create_skill(..., overwrite=true)`, and
  revision-bound `delete_skill` provide full private CRUD. Activation lasts one
  request, remains advisory, and pauses/quarantines on contrary evidence. Changes
  are live without restart; the packaged shared catalog stays read-only.
- **Add tools by asking**: drop a `BaseTool` subclass under `aeon/tools/`, declare
  its reviewed compute route in `aeon/core/tool_resources.py`, and call
  `restart_aeon`; the dynamic loader picks it up and refuses undeclared tools.
- **Fail-closed candidate execution**: `restart_aeon`,
  `verify_self_modification`, and `run_self_benchmark` auto-derive the canonical
  source root, but on this host they stop before candidate import/copy/process
  launch. Re-enabling requires an actively probed masked-home dependency sandbox;
  model-driven verification additionally requires a preconnected transport bound
  to the exact Fleet-issued loopback endpoint. User-systemd property readback and
  a TCP port-only rule are not accepted as proof. The tools never fall back to
  package installation, direct Python, or manual re-exec.

- **Durable git checkpoints.** Once the isolation latch is available, an accepted
  `restart_aeon` first tags the working tree as
  a recoverable checkpoint (`aeon/core/checkpoint.py`), capturing tracked *and*
  untracked files. Restores reconcile modifications, deletions, and additions so
  the `aeon/` package matches the checkpoint exactly, and never touch files
  outside it. Checkpoints persist as a diffable lineage (not a one-shot tarball).
- **Boot handshake (no bricking).** After the isolation latch admits a restart,
  the smoke/unit gates run the new code as a
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
  experiment **ledger**. `run_self_benchmark` exposes that score only where the
  masked-home candidate boundary is actively proved; on this host it returns the
  isolation blocker before candidate execution. Deterministic and model-driven
  task definitions remain dormant in the same registry.

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
`0600`. Aeon's browser tools supply that credential automatically. The
operator-only `aeon/scripts/browser_service.py` helper records the exact local
image ID and deterministic browser-source digest, then creates one random,
receipt-bound, CPU-only container. It never discovers containers by name or list,
and later calls can inspect, start, or restart only the exact receipted ID. The
authenticated health response repeats that random service identity, so a process
merely occupying port 8030 is not accepted as the browser backend. CPU-only
launches use the coordinator-recognized exact `CUDA_VISIBLE_DEVICES=void`
sentinel; the independent ordinal/AMD sentinels remain `-1`, and closed device
policy plus the absence of accelerator device mounts remain the enforcement
boundary.

Model-directed navigation accepts public HTTP(S) only. At the application layer,
the service validates all DNS answers for every document, subresource, WebSocket,
and redirect URL before allowing the request; loopback, LAN, link-local, metadata,
credential-bearing, file, data, Chrome-internal, and FTP destinations are refused.
Service workers are disabled so they cannot bypass interception, and every page
is revalidated before its DOM, text, or screenshot is exposed. Direct
image/video/PDF capture uses no redirects or inherited/upstream proxy, pins the
connection to the already-validated public IP, and streams into a new private
file under fixed byte caps; helper fallbacks run CPU-only in a new process group
with bounded diagnostics and whole-group timeout cleanup. Chrome still owns its
browser-network DNS connection after application admission; deployments whose
threat model includes a malicious authoritative DNS server need a network-layer
public-egress filter or pinning proxy in addition to these application checks.

The human-v6 browser contract also provides semantic `browser_find` across
off-screen content, iframes, and open shadow roots; grounded visible page text on
every observation; explicit wait success/timeouts; privately staged workspace
uploads; and bounded HTTP/XHR/console failure events. A minimal window manager
gives headed Chrome a real 1920×1080 desktop instead of Xvfb's small default
window. It also supports screenshot-coordinate interaction for canvas/map/remote
desktop UIs, explicit native-dialog responses, and structured `browser_extract`
for forms, tables, links, and page text. Named persistent profiles can be selected
with `--browser-profile NAME` or `AEON_BROWSER_PROFILE=NAME`. The launcher requires
the exact human-v6 authentication/API/source labels and immutable image receipt,
so an older authenticated container cannot silently omit these capabilities.

After changing reviewed browser-service source, an operator builds and records
the exact image, then replaces only the immutable receipted service:

```bash
python3 -m aeon.scripts.browser_service build-image
python3 -m aeon.scripts.browser_service replace-current
```

The replacement command uses the private lifecycle lock, stops only the exact
receipt-bound container ID, archives that receipt, retains the stopped container,
and provisions the recorded image. It never lists, discovers, removes, or adopts
a container.

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
- **Proxy boundary.** The receipted service refuses inherited or upstream proxy
  variables. This keeps public-destination admission and direct media IP pinning
  on one local resolver/transport boundary instead of delegating DNS to an
  unverified proxy.
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

## Self-modification

Aeon can edit its own source. Automatic candidate verification, benchmarking, and
restart are currently unavailable on this host because its unprivileged isolation
stack cannot prove both masked-home imports and exact-destination loopback model
transport. Those tools fail before copy, subprocess, checkpoint, or re-exec; they
must not be bypassed with shell/Python/package-install workarounds. An operator can
review and deploy the preserved edits through the normal service workflow. All
additions remain portable — Aeon runs from any project directory and must not
depend on its own source being in the workspace.

## Testing

```bash
./run_tests.sh                     # smoke test + unit tests (no GPU/model needed)
python3 -m aeon.smoke_test         # imports, tool discovery, syntax
python3 -m aeon.tests.test_core    # pure-logic unit tests
```

These are the same gates the restart path runs, so they are the fast way to
confirm a change is safe before going live.
