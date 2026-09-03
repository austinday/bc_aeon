# Qwen runtime capability promotion contract

`aeon/core/data/qwen_runtime_capabilities.json` is an adapter-internal release
manifest, not a live GPU inventory or an application placement API. Fleet Compute's
enabled profile manifests are the executable workload/placement authorization; the
Qwen adapter requires a matching capability here and persists its exact byte
SHA-256 and selected key in lease, runtime, and container receipts. The Python
registry contains a second exact projection, so changing JSON alone can never
authorize another host. Live occupancy and physical leases remain exclusively
coordinator-owned through Fleet Compute.

Placement iterates enabled capabilities in manifest order and every allowed
physical GPU within each capability. The capability manifest describes three
reviewed releases for the legacy runtime adapter:

- `qwen38-standard-177-local-docker`: `DAY2RTX6000PRO` physical GPU 0, local Docker,
  114688 tokens, K=3, and a 48.7 GiB measured plan. `DAY2RTX6000PRO` GPU 1 remains
  quarantined and is not in the capability.
- `qwen38-compact-178-128k`: either coordinator-safe physical GPU on `DAY2XRTX5000`,
  remote Docker, 131072 tokens, K=3, 8 sequences, an 8k scheduler batch, and a
  41.25 GiB budget. Its distinct packaged host receipt is exact-hash-bound.
- `qwen38-compact-180-128k`: either coordinator-safe physical GPU on `DAY2XRTX5000PRO-2`,
  remote Docker, 131072 tokens, K=3, 8 sequences, an 8k scheduler batch, and a
  41.25 GiB budget. The packaged receipt
  `aeon/core/data/qwen38_rtx5000_128k_release_receipt.json` is SHA-256-bound
  into the capability. Full raw reports remain owner-local under the corresponding
  `insights/` run directory and are intentionally excluded from source publication.

## Current production profile contract (2026-08-25)

The logical Fleet service is `aeon-qwen38-standard`, with a global reviewed
ceiling of two replicas. Its compatible least-busy pool has two profile lanes:
`DAY2RTX6000PRO` local Docker at variant priority 10 and lane capacity one, followed by the
generic compact-worker profile at variant priority 20 and lane capacity two.
The compact profile considers `DAY2XRTX5000` before `DAY2XRTX5000PRO-2`, then resolves the matching
host-qualified capability and receipt. Fleet therefore prefers the reviewed
`DAY2RTX6000PRO` GPU 0 release, fills any remaining target from a coordinator-approved
compact worker, and can replace a preempted worker lane without exposing physical
GPU identity to consumers. `DAY2RTX6000PRO` GPU 1 remains quarantined and is never a
candidate.

The pool contract uses the common 114688-token service floor and the same exact
model/artifact identities. Both compact releases can serve 131072 tokens and
eight sequences, but those extra limits are not advertised as a guarantee of the
mixed pool; the `DAY2RTX6000PRO` release remains the conservative one-sequence K=3 runtime.
The router admits only READY endpoints belonging to this exact declared pool and
selects the endpoint with the fewest active requests.

The experimental `aeon-qwen38-fast-180` profile and its promotion evidence remain
disabled. They do not participate in placement or routing and cannot silently
replace either reviewed pooled lane.

Short conversational turns use Aeon's deterministic low-latency path: greetings,
brief acknowledgements, and ordinary short questions skip the auxiliary skill-router
model call and request low reasoning effort. Recovery evidence, images, implementation,
debugging, architecture, security, research, and other complex-task signals are checked
first and retain the deeper adaptive reasoning path.

The remote-Docker adapter is implemented: immutable source staging, worker-side
model/image/source verification and final ACL/resource gate, exact container
receipt, central exact-PID heartbeat, nonce-bound loopback tunnel, crash-safe
stop/release, and `DAY2RTX6000PRO`-owned re-admission. Adapter availability alone does not
enable an unreviewed worker; the current `DAY2XRTX5000` and `DAY2XRTX5000PRO-2` placements are usable
only because each also has its own promoted host receipt and matching enabled
capability.

## Optimization batch lanes do not authorize future serving changes

`aeon-qwen38-full-gdn-quant` is a hash-bound batch profile for one shardwise
canary conversion on `DAY2XRTX5000PRO-2`. It stages the exact 55.6 GB ARA BF16 source from
`DAY2RTX6000PRO`, a pinned ModelOpt 0.46 wheel, and a 95 KB offline activation-scale
template; it may create at most 24 GB of output before settling the complete
artifact back to `DAY2RTX6000PRO` and deleting only its attempt-owned worker scratch. The
converter regenerates 400 language projections as NVFP4, keeps `lm_head` FP8,
and preserves the vision tower, embeddings, GDN recurrence-sensitive tensors,
and native MTP tensors in BF16. Its 41.25 GiB exclusive lease is UUID/claim-bound
and leaves at least 6 GiB physical headroom.

The full-GDN artifact from this lane has now been promoted only through the exact
fast-service adapter/profile and evidence named above. Any later batch output is
still non-serving by default. Another promotion requires the normal semantic,
refusal-surface, vision, structured-output, speculative-acceptance, context, VRAM,
latency, and throughput release gates plus matching adapter/profile hashes.

## TODO: promote `DAY2XRTX6000-2` only after a distinct release

The `DAY2XRTX6000-2` row is deliberately disabled. Promotion requires all of the
following before changing either the manifest or its code projection:

1. Provision and receipt an `aday`-owned runtime (rootless container or a fully
   pinned bare environment), its transitive executables, low-priority wrapper,
   exact model/source/image identities, and bounded scratch/storage paths.
2. Extend the reviewed transport with an exact user-owned bare-runtime adapter;
   the existing remote-Docker adapter cannot be silently applied to `DAY2XRTX6000-2`.
3. Pass the full text, vision, structured-output, native-MTP, deterministic,
   114688-context, VRAM-peak, performance, renter-preemption, and migration
   gates on `DAY2XRTX6000-2`; emit a machine-readable host/runtime release receipt.
4. Add the reviewed adapter and receipt hashes to the capability, enable it in
   both projections, add the matching hash-bound Fleet service-profile variant,
   and add hermetic mutation and preemption tests. After a reconciled Fleet reload,
   the broker may select that variant and hand the adapter only its exact preleased
   host/GPU identity.

## `DAY2XRTX5000` compact release is qualified in the capability registry

The exact image and model were release-staged on `DAY2XRTX5000`. The host-specific
release passed, and the source tree enables the distinct
`qwen38-compact-178-128k` key in both capability projections. Its exact packaged
receipt SHA-256 is
`fef559cd0b88506b7b0b29f12cd6c1fdee8b525fa2962358c16048529804f13d`.
The production generic compact-worker profile is enabled with `DAY2XRTX5000` ahead of
`DAY2XRTX5000PRO-2`; the bounded `DAY2XRTX5000` release-gate profile is disabled and retired after
exact teardown. Current capacity remains live coordinator state, not release
evidence, and a free 48 GB GPU alone never authorizes placement.

The exact raw report hashes independently recompute 15/15 schema- and
semantic-valid deterministic K=3 decisions, positive native-MTP acceptance,
115.34 median decode tokens/second, exact recall at 125,985 measured and reported
prompt tokens, and concurrency-eight throughput of 483.87 decode tokens/second
versus 82.93 serial. The long-context/batch report is schema v2 and binds the
exact benchmark script SHA-256. A private lifecycle-state hash binds the
sanitized host, image, model-manifest, source-manifest, memory, exclusivity, and
41.25 GiB budget attestation without packaging its endpoint, claim, GPU UUID,
PID, container, or filesystem identifiers. Raw reports and private lifecycle
evidence remain outside the source tree.

The evidence package now also binds the exact disabled candidate-manifest identity
used by the release gate and the sanitized ordinary-Aeon gate. That normal process
passed startup vision, emitted one exact structured `pwd` action, grounded its
truthful final response in the command receipt, and verified its own broker-ticket
release and session cleanup. This proves the normal routed semantic transport. A
coordinator READY-state
sample observed 35,376 MiB used and 13,028 MiB free under the enforced 42,240 MiB
(41.25 GiB) committed budget, with zero lease violations, zero ambiguous intents,
and the watchdog active. This is explicitly a sampled READY observation, not an
invented historical peak. The final exact teardown report then proved the Fleet
runtime stopped without error, zero active release-gate demand, an AVAILABLE/OPEN
coordinator slot with zero claims, violations, or ambiguous intents, and absence
of the exact local controller/orchestrator receipts, worker receipt/run directory,
and container. The intentionally empty canonical `DAY2RTX6000PRO` journal was preserved.

`DAY2XRTX5000` and `DAY2XRTX5000PRO-2` are qualifications of one portable compact remote-Docker
runtime contract, not separate implementations. They share the immutable image,
model manifests, source staging protocol, 131072-token limit, K=3 sampling,
41.25 GiB cap, batching limits, transport, readiness, and lifecycle adapter.
Each host must nevertheless carry its own immutable packaged qualification
receipt: the receipt validator binds the exact host and hostname as well as the
shared runtime fields and required gate/report evidence. A receipt from `DAY2XRTX5000PRO-2`
can never qualify `DAY2XRTX5000`.

Canonical image, model, and source identities remain on `DAY2RTX6000PRO`. The enabled
compact-worker profile now declares one Fleet-owned shared-cache contract for
all three: manifested source and model trees plus an OCI image-config identity.
Fleet stages only manifest-listed bytes into content-addressed worker paths,
verifies the complete payload and Fleet ownership marker before atomic publish,
and holds a runtime reference before the adapter can consume a binding. Cold OCI
preparation uses a bounded, low-priority archive transfer, exact checksum and
image-ID/config verification, then retains only a small cache receipt. A worker
that already proves the exact image takes the warm path and skips archive export,
transfer, and load. Runtime preflight revalidates the bindings before creating a
fresh container.

The worker cache has explicit decimal transfer/cold-stage bounds, a 64 GiB
managed quota, inode limits, retained free-space/inode reserves, and a 24-hour
idle TTL. Eviction is deterministic and limited to zero-reference entries whose
exact Fleet ownership and immediate inode/content token are re-proven. It never
deletes canonical `DAY2RTX6000PRO` data or Docker's global image/layer store. Aeon adds no
fixed-host pool, independent allocator, availability scanner, cache daemon, cron
job, or age-only scavenger.

Capability and profile promotion are complete in the source tree. The enabled
generic compact-worker profile is the sole production remote serving lane, with
ordered `DAY2XRTX5000` and `DAY2XRTX5000PRO-2` placements and independently validated receipts. The
legacy `DAY2XRTX5000PRO-2` profile remains disabled only as an exact historical
recovery/teardown identity. Runtime availability is still reconciled against live
renter, ACL, lease, storage, host, and network state; promotion never guarantees
that either worker is available now.

Automatic source/model/OCI preparation and bounded worker-cache collection are
part of the reviewed Fleet profile contract. They do not expand qualification:
`DAY2XRTX5000` and `DAY2XRTX5000PRO-2` still require their own exact release receipts, `DAY2XRTX6000-2` remains
disabled, and unavailable or ambiguous storage/runtime evidence remains durable
waiting or a fail-closed error rather than permission to use another host.

The old `gate_qwen38_rtx5000.py start` path fails before coordinator or runtime
work. The durable Fleet release-gate profile used for `DAY2XRTX5000` qualification is now
disabled and retired. Its `stop` subcommand is retained solely for exact teardown
of a historical receipt under the operator recovery runbook; it is not a launch
path.
