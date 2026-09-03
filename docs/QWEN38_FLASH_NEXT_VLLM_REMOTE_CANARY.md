# Qwen3.8 Flash-Next vLLM `DAY2XRTX6000-2` GPU1 canary

This is a disabled parallel qualification lane. Fleet Compute is the only
allocator and may place it only on `DAY2XRTX6000-2` physical GPU 1. Its current
operator route is `192.168.8.112`; Fleet may retain a legacy internal coordinator
host selector, which is not an SSH address. Physical GPU
0 is outside the executable placement contract and must remain untouched when
rented or otherwise unavailable.

The lane uses the exact local canary checkpoint, NVFP4 vLLM image, MTP3 runtime,
semantic suite, CUDA placement proof, and promotion thresholds. It stages the
421-file checkpoint plus the single-platform OCI archive from canonical `DAY2RTX6000PRO`
storage into a unique Fleet attempt below
`/home/aday/.local/state/fleet-compute/runs`. Every source file, checkpoint file,
manifest, and archive is verified against the release-bound SHA-256 identities
before Docker load or GPU execution.

The remote wrapper is independently hash-bound, while the shared source-manifest,
runtime-contract, and worker identities are byte-identical to the native allocator
local canary. The disabled profile intentionally retains its reviewed 192 GiB
available-memory and commit-headroom floors; a host below either floor remains
ineligible rather than silently weakening admission.

## Cold staging budget

- Model ceiling: 137,000,000,000 bytes
- OCI archive ceiling: 8,700,000,000 bytes
- Source/assets ceiling: 20,000,000 bytes
- Total stage ceiling: 145,720,000,000 bytes
- Runtime-growth ceiling: 64,000,000,000 bytes
- Retained worker free-space reserve: 20,000,000,000 bytes
- Admission floor: 230 GB free plus the inode, RAM, commit, `/dev/shm`, and GPU
  checks in the profile
- Transfer limiter: one renter-yielding stream at 100,000,000 bytes/second

The transfer-only lower bound is about 24.3 minutes at the limiter. A realistic
cold preparation estimate is 35–60 minutes after lease acquisition because both
ends also verify the 421-file model manifest and the OCI archive. These are
planning estimates, not readiness claims.

## Settlement and cleanup

The remote worker writes the same `qualification.json` and evidence closure as
the local lane. Fleet copies only the explicit output, MTP-arm evidence, status,
and bounded supervisor logs back to the canonical `DAY2RTX6000PRO` artifact directory,
generates `SETTLED.sha256`, re-runs the shared promotion validator, and records a
promotion-compatible settlement receipt. Cleanup remains fail-closed until the
terminal status, process absence, two exact task-container absences, filesystem,
non-symlink tree, and durable unguessable ownership token are all revalidated.
Only the exact attempt directory is then removed; shared caches, Docker images,
canonical `DAY2RTX6000PRO` data, and unrelated worker paths are never cleanup targets.

Promotion still requires at least 120 token/s single-stream decode after prefill,
at least 490 token/s C4 aggregate completion throughput, positive MTP drafting
and acceptance with causal equivalence, all semantic gates, all compute weights
on CUDA, PLE as the sole CPU model component, and the 88 GiB cap plus 6 GiB
physical reserve. This lane must remain disabled until a deliberate Fleet reload
and separately authorized live canary submission.
