# Fleet Compute integration

Aeon 0.2 has a consumer-side integration for the developing Fleet Compute
broker. It does not install, start, or configure that broker.

## Runtime selection

`AEON_COMPUTE_BACKEND=auto` is the default:

1. No broker socket: use Aeon's existing coordinator-managed Qwen lifecycle.
2. Healthy owner-only broker socket plus an enabled
   `aeon-qwen38-standard` service profile: acquire and renew a demand ticket,
   wait durably for compute, and use its loopback OpenAI-compatible endpoint.
3. Present but unsafe/unreachable broker socket, duplicate/malformed profile, or
   broker runtime evidence without a matching enabled profile: fail closed.

`AEON_COMPUTE_BACKEND=broker` requires the broker path. The temporary migration
escape hatch `AEON_COMPUTE_BACKEND=coordinator` selects the existing lifecycle.
`AEON_FLEET_SOCKET` and `AEON_FLEET_PROFILE` override only the socket/profile
identity; applications still cannot pass host, GPU, claim, or launch commands.

## Profile handoff

The packaged
`aeon/core/data/fleet_compute_aeon_qwen38_profile.disabled.json` is a disabled,
machine-readable broker draft anchored to Aeon's `.177` capability. Aeon's
coordinator compatibility lifecycle separately supports the released `.180`
128k remote-Docker capability. The broker draft must remain disabled until the
Fleet Compute project has all of the following:

- a reviewed `aeon-qwen38-runtime-v1` adapter registered through its documented
  entry-point mechanism;
- exact artifact, image, source, launch, PID, readiness, heartbeat, stop, and
  absence proofs equivalent to Aeon's current Qwen runtime receipts;
- startup heartbeat behavior that satisfies the fleet's 15-minute maximum even
  during the bounded model-load/warmup transaction;
- tests using fake coordinators/adapters plus a separately authorized,
  lease-bound live release receipt;
- an operator-approved broker service exception in the global fleet policy.

Do not copy the draft into `~/fleet_compute/profiles.d`, add its manifest hash,
or enable it as part of an ordinary Aeon install. Enabling a profile is executable
fleet authorization and belongs to the broker rollout.

## Endpoint contract

Aeon accepts only a credential-free `http://127.0.0.1:PORT[/v1]` or IPv6-loopback
equivalent from the broker. Ticket IDs must match the broker's `fd-` plus 32
lowercase hex format. If a replacement runtime changes the endpoint during an
Aeon process, the process fails visibly and asks for a restart rather than
silently rebinding an already-created model client.
