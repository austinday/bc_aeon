# Qwen runtime capability promotion contract

`aeon/core/data/qwen_runtime_capabilities.json` is an authorization manifest,
not a live GPU inventory. Its exact byte SHA-256 and selected capability key are
persisted in reservation, lease, runtime, and container launch receipts. The
Python registry contains a second exact projection, so changing JSON alone can
never authorize another host. Live occupancy and placement remain exclusively
coordinator-owned.

Placement iterates enabled capabilities in manifest order and every allowed
physical GPU within each capability. Two releases are enabled:

- `qwen38-standard-177-local-docker`: `.177` physical GPU 0, local Docker,
  114688 tokens, K=3, and a 48.7 GiB measured plan. `.177` GPU 1 remains
  quarantined and is not in the capability.
- `qwen38-compact-180-128k`: either coordinator-safe physical GPU on `.180`,
  remote Docker, 131072 tokens, K=3, 8 sequences, an 8k scheduler batch, and a
  41.25 GiB budget. The packaged receipt
  `aeon/core/data/qwen38_rtx5000_128k_release_receipt.json` is SHA-256-bound
  into the capability; the full receipt and raw reports live under
  `insights/qwen38_rtx5000_128k_20260822T1822/`.

The remote-Docker adapter is implemented: immutable source staging, worker-side
model/image/source verification and final ACL/resource gate, exact container
receipt, central exact-PID heartbeat, nonce-bound loopback tunnel, crash-safe
stop/release, and `.177`-owned re-admission. Its presence does not enable a
worker; the corresponding host/profile receipt still must be promoted below.

## TODO: promote `.179` only after a distinct release

The `.179` row is deliberately disabled. Promotion requires all of the
following before changing either the manifest or its code projection:

1. Provision and receipt an `aday`-owned runtime (rootless container or a fully
   pinned bare environment), its transitive executables, low-priority wrapper,
   exact model/source/image identities, and bounded scratch/storage paths.
2. Extend the reviewed transport with an exact user-owned bare-runtime adapter;
   the existing remote-Docker adapter cannot be silently applied to `.179`.
3. Pass the full text, vision, structured-output, native-MTP, deterministic,
   114688-context, VRAM-peak, performance, renter-preemption, and migration
   gates on `.179`; emit a machine-readable host/runtime release receipt.
4. Add the reviewed adapter and receipt hashes to the capability, enable it in
   both projections, and add hermetic mutation and preemption tests. The
   placement loop will then try its exact host/GPU selectors after `.177`.

## `.178` compact remains disabled

The `.178` row remains disabled because it lacks safe staging space, has a stale
image, and has no exact host release receipt. A free 48 GB GPU does not authorize
placement. Promotion requires a separate immutable receipt covering safety,
semantic, vision, structured output, long context, VRAM, transport, and
preemption gates; `.180` evidence cannot be copied across hosts.
