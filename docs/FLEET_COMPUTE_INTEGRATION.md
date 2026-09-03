# Fleet Compute integration

Aeon uses the authorized Fleet Compute broker for Qwen and ComfyUI. It does not
create a second allocator.

## Runtime selection

`AEON_COMPUTE_BACKEND=broker` is the production default; `auto` is a broker-only
alias. A missing/unsafe broker, duplicate or malformed profile, or runtime
evidence without a matching enabled profile fails closed.
`AEON_FLEET_SOCKET` and `AEON_FLEET_PROFILE` override only the socket/profile
identity; applications still cannot pass host, GPU, claim, or launch commands.

## Profile handoff

Fleet Compute has an enabled, hash-bound, two-replica
`aeon-qwen38-standard` serving pool across preferred `DAY2RTX6000PRO` GPU 0 and one generic
compact-worker lane. That worker profile considers `DAY2XRTX5000` before `DAY2XRTX5000PRO-2`; each
placement resolves its own enabled, host-qualified capability and receipt. The
local and compact-worker profiles share one explicit serving-pool identity while
retaining distinct lane ceilings and resource contracts. `aeon-comfyui` remains
the image/edit service on `DAY2RTX6000PRO`. Video uses the separate logical service
`aeon-video-comfyui`: its priority-10 lane reuses the exact local runtime when
`DAY2RTX6000PRO` GPU 0 is safe, while its priority-20 lane stages the reviewed image and
model bundle to a coordinator-selected 48 GB card on `DAY2XRTX5000` or `DAY2XRTX5000PRO-2`. This avoids
a circular wait behind the always-on exclusive Qwen service without weakening
either lease. `DAY2RTX6000PRO` GPU 1 remains quarantined; `DAY2XRTX6000-2` and the experimental
fast `DAY2XRTX5000PRO-2` variant remain outside production Qwen serving. Each release
requires:

- reviewed `aeon-qwen38-runtime-v1`, `aeon-comfyui-runtime-v1`, and
  `aeon-video-comfyui-runtime-v1` adapters
  registered through the documented entry-point mechanism;
- exact artifact, image, source, launch, PID, readiness, heartbeat, stop, and
  absence proofs equivalent to Aeon's current Qwen runtime receipts;
- startup heartbeat behavior that satisfies the fleet's 15-minute maximum even
  during the bounded model-load/warmup transaction;
- tests using fake coordinators/adapters plus a separately authorized,
  lease-bound live release receipt;
- an operator-approved broker service exception in the global fleet policy.

Enabling or changing a profile is executable fleet authorization and belongs to
the Fleet Compute rollout, not an ordinary Aeon install.

The final one-RTX Flash-Next vLLM promotion is an atomic replacement, not an
additional pooled replica.  Its disabled production profile remains inert until
the exact uncensored NVFP4+MTP checkpoint, OCI archive, source set, promotion
binding, and final canary receipt are hash-bound.  That receipt must prove all of
the following on the production RTX PRO 6000 shape: at least 120 token/s
single-stream decode measured after prefill; positive engine-native MTP drafted
and accepted counters; semantic/tool/reasoning/multimodal health; 128K context or
the explicitly selected largest context actually exercised by the canary; and at
least 6 GiB of physical VRAM reserve under dense independent sampling.  Aggregate
C4 throughput is diagnostic evidence, not a promotion floor.

After those proofs are reviewed, one reconciled Fleet reload must disable the
compact-worker profile and enable the exact Flash-Next vLLM profile together;
enabling both is intentionally rejected because their serving-pool identities
are incompatible.  The authenticated service policy is then set to one replica
with `qwen38-flash-next` preferred.  Routing remains through the unchanged
logical `aeon-qwen38-standard` endpoint, so Aeon needs no concrete-profile
override.  If the new lane does not reach semantic readiness, loses its exact
process/artifact identity, or regresses any release gate, rollback is the inverse
reviewed transaction: disable the Flash-Next lane, re-enable the unchanged
compact-worker profile, reload Fleet after reconciling active work, and set the
policy back to one reviewed `qwen38-27b` replica.  Never force-release a claim to
perform either switch.

The `DAY2XRTX6000-2` production-K3 speed qualification is a separate disabled batch
profile, `aeon-qwen38-production-k3-v026-canary-179`; it is not a serving lane
and cannot join `aeon-qwen38-standard`. Its bare dev1141 engine is bound both to
the exact canonical `DAY2RTX6000PRO` archive and to a compact extraction receipt generated
from every member of that archive: 81,903 paths, 73,103 regular files,
8,796 directories, four exact reviewed Python-venv symlinks, and 8,460,059,803
regular bytes. Preflight recomputes the globally path-sorted closure, hashes every
regular file through a no-follow stable descriptor, rejects unexpected symlinks,
hard links, special inodes, cross-filesystem entries, mutable ownership/modes,
and path-set drift, and compares the result to the profile-bound closure digest.
It separately binds the exact Python executable digest, size, version, cache tag,
and SOABI. Selected vLLM sentinels and torch/vLLM/CUDA import assertions remain
additional semantic checks, not substitutes for the full closure.

The separate `aeon-qwen38-flash-next-build` batch profile is also shipped
disabled. It is pinned only to `DAY2XRTX6000-2` and combines immutable source staging,
BF16 feature extraction and rank-four output-head behavioral tuning, assembly
of the official BF16 transformer with only the official FP8 PLE table, and
routed-expert-only ModelOpt NVFP4 conversion. Its 88 GiB UUID-bound exclusive
lease, 510 GB peak-safe disk admission floor, 165 GiB available-RAM floor, and
154 GiB commit-available floor are one build authorization. The trainer's exact
review map uses 74.87 GiB GPU, 144.26 GiB CPU, and 63.61 GiB disk under a 152 GiB
CPU budget, then rechecks both live RAM and eight-GiB commit headroom after
deriving that map. Source shards use digest-bound atomic resumable partials,
and three bounded attempts preserve completed downloads within an attempt while
restarting the non-checkpointed feature cache after preemption. Exact output
settlement/worker-cleanup receipts remain mandatory. This is not a serving lane.
SGLang text/image/video and MTP qualification is a separate Fleet-backed `DAY2RTX6000PRO`
release gate against the pinned official image.

Only this disabled dev1141 K3 variant declares
`enable_per_request_metrics=true`. Its preflight invokes the exact pinned engine's
API-server help with no visible GPU and refuses launch unless
`--enable-per-request-metrics` is present; its final server argv then contains
that flag. Existing v0.23/standard variants keep the property false and never
receive the flag. This preserves the single-axis qualification while allowing
the benchmark's usage-bearing requests to record server-scoped timing metrics.

The `DAY2XRTX5000` release passed the full qualification suite and now resolves as
`qwen38-compact-178-128k` with its own exact packaged release receipt. It shares
the portable compact remote-Docker contract used by `DAY2XRTX5000PRO-2`, but the two host
capabilities remain independently qualified: matching model and image identities
do not make either receipt transferable. The production
`aeon-qwen38-compact-workers` profile is enabled with `DAY2XRTX5000` then `DAY2XRTX5000PRO-2` as its
ordered placements. The one-host `DAY2XRTX5000` qualification profile and the legacy
host-specific `DAY2XRTX5000PRO-2` serving profile are disabled; neither is an additional
production lane or allocator.

The remote adapter now consumes Fleet's single shared artifact-cache contract.
After admission, Fleet reference-counts and atomically stages three exact objects
from canonical `DAY2RTX6000PRO`: Aeon's manifested source allowlist, the manifested Qwen
model tree, and the exact OCI image-config identity. A cold worker receives a
rate-limited archive that is checksum-verified, loaded, and re-inspected before
the temporary archive is reduced to a small Fleet receipt. If that exact image is
already installed, the adapter proves its ID, logical size, and configuration and
creates only the receipt; it does not transfer or load an archive. Every launch
still creates a fresh, receipt-bound container.

The managed Qwen cache lives only below the profile's explicit worker root. Fleet
records active references and last use, applies the 24-hour idle TTL and 64 GiB
quota only to proven zero-reference entries, and revalidates an opaque inode/
ownership token immediately before deletion. Attempt teardown never evicts a
shared entry. Neither reconciliation nor the adapter deletes canonical `DAY2RTX6000PRO`
artifacts or anything in Docker's global store, and there is no separate cache
daemon, fixed-host allocator, or age-only scavenger. `DAY2XRTX5000` and `DAY2XRTX5000PRO-2` remain the
only qualified remote-Docker workers; automatic staging does not qualify `DAY2XRTX6000-2`
or make a matching image/model transferable across host release receipts.

The video worker lane uses the same Fleet cache protocol under its own root and
quota. It stages one exact untagged ComfyUI OCI archive plus the H3 and LTX files,
validates every payload and Docker image ID, then starts a unique claim/runtime-
labeled container through `fleet-low-priority`, pinned to the lease UUID and 40 GB
cap. Only `DAY2XRTX5000` and `DAY2XRTX5000PRO-2` are enabled because those two release workers expose
the owner-authorized Docker client; `DAY2XRTX6000-2` is not inferred compatible merely from
its GPU size. A private loopback SSH tunnel is the only endpoint returned to Aeon.
Comfy output stays in attempt-owned worker scratch until Aeon downloads and
validates it, after which Fleet proves container/tunnel absence and removes only
that exact scratch tree. Shared cached inputs remain reference-counted and bounded.

Replica targets are durable across renter preemption and broker restarts. If a
remote launch creates a container but fails before Fleet commits its process
identity, reconciliation first binds the immutable container ID from the exact
worker receipt, then performs one controller-locked ID/PID/claim-verified cleanup.
Only after worker receipt, container, scratch, and tunnel absence are proven does
Fleet release the cooperative claim and retry the missing replica on compatible
coordinator-approved capacity. Any identity, controller, or absence ambiguity
remains quarantined and cannot authorize a replacement.

## Change propagation

Nexus and Fleet Compute import `/home/aday/NexusAgentDashboard/bc_aeon` directly. New processes see
ordinary Python edits; existing agents and long-running services do not reload
already-imported code. Restart the affected Nexus-managed agent for Aeon core/tool
changes, restart Nexus for imported remote/instruction-layer changes, and restart
Fleet at a reconciled maintenance point for adapter/profile/entry-point changes.
Refresh the editable Aeon install only when dependencies, console scripts, or entry
points change. The complete upgrade matrix and verification procedure are in
`/home/aday/NexusAgentDashboard/fleet_compute/docs/INTEGRATED_OPERATIONS.md`.

## Endpoint contract

Aeon accepts only a credential-free `http://127.0.0.1:PORT[/v1]` or IPv6-loopback
equivalent from the broker. Whitespace, ASCII controls, URL parameters, and
non-canonical spellings are rejected before the endpoint is retained. Ticket IDs
must match the broker's `fd-` plus 32 lowercase hex format. Fleet returns one stable loopback request-router endpoint
for the shared Qwen service. A running full Aeon renews its ticket in the
background and stages a changed endpoint; the next foreground compute guard
rebinds both language and vision clients. Replica membership changes behind the
stable router require no Aeon restart. A broker restart may choose a new loopback
port, which follows the same renewal/rebind boundary.

The OpenAI-compatible client's no-proxy/no-redirect HTTP transport also invokes
the owning session guard immediately before every actual local request, including
support calls and retries. It rechecks the exact origin and API path both before
and after that guard. If the guard promotes the service, the old in-flight request
is refused and only a retry through the rebound client may continue; a client with
no ticket guard cannot send local model traffic.

Each Qwen demand snapshot is identity-bound, not treated as a loose readiness
hint. Acquire, status, renewal, and release responses must repeat the exact
opaque ticket, consumer, canonical profile/logical-service identity, compatible
state, compute state, endpoint semantics, and a sorted bounded set of the
concrete READY runtime profiles behind that endpoint. The profile set contains no
runtime, process, host, claim, or GPU identity. Aeon maps it to explicit
Flash-Next, 27B fallback, or mixed-pool status text; the logical service ID and
legacy OpenAI wire alias are never used as proof of the concrete model artifact.
Every lane in that reviewed pool accepts the legacy
`Qwen3.8-27B-ARA-NVFP4-MTP` token as a compatibility-only API name, including
the Flash-Next primary. The router sends requests only to READY lanes for the
owner-selected priority model. Fleet fills only that model family's reviewed lane
ceiling while one of its lanes is STARTING or READY. It attempts the compact
family only after all eligible priority-model placements refuse admission;
least-busy balancing applies within the selected model family, never across
primary and fallback. When a priority lane later becomes available, new traffic
moves to it and the fallback drains through the ordinary bounded idle lifecycle.
This preserves a stable agent model token while still allowing reviewed failover;
user-facing identity continues to come from the concrete runtime profiles.
Logical-service variants may carry different nonempty, lane-specific purpose
descriptions; compatibility is established by the reviewed project, service ID,
routing policy, enablement, and exact advertised variant set rather than by
requiring descriptive prose to be identical.
A concrete requested deployment
profile may canonicalize to its reviewed logical service on acquisition; that
returned identity is then immutable for the ticket. Malformed release evidence
never clears local ownership, so cleanup can retry only that exact demand and the
process-exit fallback gets another attempt. Conversely, a valid-looking ID in an
acquisition response is not ownership: Aeon binds or releases it only after the
exact consumer and reviewed logical-service identities match. An ambiguous or
cross-wired ID is never deleted; Fleet's bounded TTL expires it safely.

The service-ticket renewal response is also the current sanitized compute proof.
Every successful renewal republishes `allocated` or `waiting_for_compute` to the
active Aeon presence manifest, refreshing `compute_updated_at`. A broker transport
outage retries with bounded backoff only until the locally tracked expiration of
the last broker-validated lease and publishes a truthful reconnecting state; a
successful retry refreshes the lease and resumes the existing session. Expired or
inactive demand, malformed or identity-drifting evidence, and any transport outage
that reaches the lease deadline publish `unavailable` and latch an error
immediately. This keeps Nexus's deliberately bounded allocation evidence current
without exposing the ticket, endpoint, claim, process, or device identity, while a
brief reconciled broker restart does not poison every live agent session.

## Bounded sub-agent ownership

Every bounded sub-agent using local Qwen owns a separate expiring Fleet service
ticket with consumer identity `aeon/sub-agent/<agent-id>`. The child validates the
same broker-only backend and enabled service contract as a full Aeon process,
replaces the principal's inherited URL with its broker-returned loopback endpoint,
and attaches a foreground compute guard. Multimodal/vision calls follow that same
child endpoint, including broker endpoint promotion; they never silently reuse the
principal's ticket-affine route.

Normal completion, initialization or execution errors, catchable termination, and
the bounded watchdog converge on closing the exact child ticket. Admission waits
remain heartbeated and cancelable. Each new child also owns a canonical-UUID user
systemd scope inside a unique flat leaf slice. Its schema-2 receipt pins both units
by exact name, `InvocationID`, `ControlGroup`, and `ControlGroupId`, and revalidates
the fixed CPU-only scope policy before any signal. Recursive `populated` readback
from that exact slice's `cgroup.events` is the descendant-liveness authority; Aeon
does not enumerate processes or systemd units. Principal-initiated stop sends
SIGTERM to that exact slice, allows 30 seconds for child-ticket cleanup, then
revalidates the complete receipt immediately before an exact-slice SIGKILL. Empty
slices are stopped and revalidated absent. Any malformed readback, identity drift,
or unit/cgroup disagreement refuses signaling. Legacy schema-1 sessions remain
readable and may signal their revalidated live wrapper group, but once that leader
exits Aeon refuses descendant escalation if the numeric group still exists.

The launcher generates and exports
`AEON_CPU_SANDBOX_SLICE=aeon_subagent_<agent-uuid-hex>.slice` only after scrubbing
inherited authority. A nested generic shell uses a cryptographically unique
transient user service with the exact protected `Slice`; it rejects caller/model
slice overrides and verifies both the service `Slice` and cgroup path under the
receipted leaf before opening its file gate. Its exact unit, MainPID, cgroup, and
InvocationID are durably receipted before the requested shell runs, and every
later stop/recovery action revalidates unit plus InvocationID instead of signaling
a numeric workload PID. Inherited GPU selectors, claims, limits, tickets, leases,
and model/media endpoints are removed. The replacement CPU-only CUDA selector is
exactly `CUDA_VISIBLE_DEVICES=void`, which the coordinator classifies as disabled;
ordinal and AMD visibility sentinels remain `-1`. This environment scrub is
defense in depth beside the closed device policy and device-path denial. Generic
services deny all socket creation;
network-capable browser/provider tools and Fleet adapters own separate reviewed
boundaries.

If a process is destroyed with an uncatchable signal or the broker cannot confirm
release during forced cleanup, only that ticket's bounded TTL remains; the wrapper
never guesses at or releases another consumer's demand. Subscription/API provider
configurations do not create local Fleet demand.

## Tool compute-route contract

Every executable `BaseTool` name is declared in exactly one route in
`aeon/core/tool_resources.py`. Construction alone grants no compute authority:
both worker registration and the immediate execution boundary revalidate the
declaration, and the rendered tool catalog shows the route to the model before it
chooses a call. An undeclared or runtime-modified route is blocked.

- `local_cpu` tools use ordinary bounded host/CPU resources and never infer GPU
  availability.
- `active_model` tools reuse the current model provider. For local vLLM or
  llama.cpp, the worker calls the owning Fleet ticket guard immediately before
  execution; provider APIs do not create owner Fleet demand.
- `fleet_service` image and edit calls acquire a fresh `aeon-comfyui` demand;
  video calls acquire `aeon-video-comfyui`. They use Aeon's
  owner-ACL-validating Unix-socket client,
  accept only the exact ticket/consumer/service and a credential-free loopback
  endpoint, renew during waits and active work, and require exact inactive
  release proof in `finally` cleanup. Initial Comfy acquisition follows the same
  ownership-before-release rule, so a cross-wired consumer or service ID is left
  to bounded TTL rather than risking deletion of another demand. Video generation selects the complete
  MiniMax H3 NVFP4 audiovisual stack only when every exact local component is
  present; otherwise it fails before acquiring compute. The reviewed LTX 10Eros
  stack is a deliberate capability fallback, never a placement fallback. The
  obsolete local registry path is gone.
- `fleet_batch` exposes a separate typed recipe catalog, durable submission, and
  owned-job status path. The model never receives Fleet's generic profile/payload
  API: each callable recipe binds one enabled reviewed batch profile, exact
  project, and closed payload in source. Recipe eligibility is also checked
  against the active owner goal. A general request such as “make a useful model
  on our GPUs” currently receives an empty catalog because no general-purpose
  Hugging Face build adapter has passed the profile/storage/release contract;
  policy text cannot substitute for that executable review. The agent must then
  continue a genuinely CPU-safe contribution or report the limitation once,
  never probe GPU devices, SSH, Docker, or the broker for a bypass.
  Submitted jobs use ordinary standard demand, an agent-bound idempotency key,
  and an owner-private per-agent receipt. Status refuses jobs without that exact
  receipt. Queued, starting, running, settling, and cleanup states are typed
  pending evidence. Continuous mode waits 30 cancellable seconds before a fresh
  status cycle instead of immediately resubmitting or restarting discovery.
- `fleet_child` calls create a bounded process whose local model owns its own
  demand and endpoint; they do not reuse the principal ticket.
- `dynamic_command` calls are CPU-only shell boundaries, not allocators. Their
  execution guard rejects direct GPU/coordinator/device/lease and process-escape
  forms before process or job-state creation; GPU work must be submitted to an
  enabled Fleet service or reviewed batch adapter.
- `nexus_lifecycle` asks the existing authenticated Nexus lifecycle to create a
  standalone agent, which then owns its own Fleet session when local Qwen is
  selected.
- `external_provider` calls use the configured provider; any disclosure review
  performed by a local primary model still crosses its current Fleet guard.
- `host_service` calls use an already-provisioned, loopback-only CPU dependency.
  Browser and search tools have no Docker authority. Their operator helpers use
  random identities, owner-private exact receipts, hard CPU/RAM/PID/`/dev/shm`
  limits, no accelerator devices or environment, and authenticated or
  identity-bound semantic health. A listener on the expected port alone is not
  accepted as service proof.

The route manifest is an admission contract, not a second allocator. It never
accepts a host, device, claim, raw availability observation, or model-supplied
environment variable as proof of compute authority.
