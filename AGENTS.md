# Aeon fleet integration

Read `/home/aday/NexusAgentDashboard/fleet_compute/docs/INTEGRATED_OPERATIONS.md` for cross-project
ownership and upgrade/reload rules; this file remains the local mandatory handoff.

- Production agents use Fleet Compute only. `AEON_COMPUTE_BACKEND=broker` is the
  default; `auto` is a broker-only alias and direct coordinator selection is refused.
- Qwen uses the logical `aeon-qwen38-standard` service; image/edit tools use
  `aeon-comfyui`, while video uses the separately profiled
  `aeon-video-comfyui` service. Consumers receive opaque tickets and loopback
  endpoints only.
- Reviewed Fleet adapters own exact launch, PID/container identity, readiness,
  heartbeat, stop, output settlement, and worker scratch cleanup. Aeon tools never
  select GPUs or release coordinator claims.
- Legacy coordinator lifecycle modules remain for adapter internals, tests, and
  explicit operator recovery. Do not call them from a normal agent/tool path.
- Never inspect, stop, or modify Vast renter containers, Vast services, device ACLs,
  pricing, the root watchdog, or unfamiliar worker data.
