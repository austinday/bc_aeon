# Nexus Project Manager lifecycle

The Project Manager is an always-present **Home terminal tab**, not an
always-running agent. Its durable registry row starts at `/home/aday` on the
orchestrator and is initially created with `status=idle` and
`desired_state=stopped`. Nexus application assembly opens only its fixed Bash
tmux shell so the initial page has a usable Home terminal. This shell start does
not start Aeon, a provider CLI, a model runtime, or a GPU coordinator lease.

`aeon.remote.project_manager` owns the stable UUID-shaped ID and protected
identity fields. `ensure_project_manager()` performs only registry `get`/`create`
operations. It accepts a concurrent creator only after re-reading and validating
the complete protected identity. A mismatch fails closed rather than adopting or
rewriting an uncertain row.

## Activation boundary

Only Nexus application assembly may open the virgin fixed shell automatically.
Ordinary list/status reads remain side-effect free. After any failure or
interruption, reopening is an explicit authenticated, CSRF-protected action;
there is no background retry loop. Starting Aeon or a provider agent is always a
separate CSRF-protected action that derives the live pane directory server-side
and validates it against `AEON_REMOTE_ALLOWED_ROOTS`.

Starting Aeon in this shell delivers the canonical Project Manager objective.
Ending it returns the same tmux tab to Bash; the tab ID, workspace, prompt
versions, and local-role versions remain durable.

Aeon's normal coordinator-bound compute path remains authoritative after resume.
No capacity is reserved merely to keep the tab warm. During the active foreground
admission loop, Aeon presence exposes `waiting_for_compute`; each coordinator call
and backoff sleep is bounded, backoff tops out at two minutes, and no lease is held
between attempts. This is not a reason to bypass the coordinator or create an
independent allocator. Ctrl-C or **End agent** cancels the wait, records compute as
`unavailable`, and exits Aeon while the durable instance remains resumable. A
process, service, or machine interruption also leaves that registry row resumable.
A dead pane must never continue to look actively queued.

Stopping the instance releases its ordinary Aeon resources and returns the row to
`idle` after its pane exits. The row itself cannot be deleted. Force-stop may end
an exact live Project Manager tmux session, but it must also preserve the row and
return to `idle` on reconciliation.

## Manager integration contract

1. Call `ensure_project_manager(store, default_model=...)` while constructing the
   instance manager, before mutation endpoints become reachable. This closes the
   name/identity race without launching anything. It is safe to re-ensure before
   returning the instance list. During Nexus application assembly only, launch
   the fixed Bash shell when the row is still virgin and idle. Never auto-launch
   an agent, retry an errored/interrupted shell, or reserve compute.
2. Merge `project_manager_public_flags(record)` into the public instance object.
3. Consult `dormant_project_manager_status()` before generic stopped-session
   reconciliation so an absent/dead pane plus `desired_state=stopped` is `idle`.
   A missing pane while desired running remains truthfully `interrupted`.
4. Call `reject_project_manager_deletion(instance_id)` before any pane lookup,
   kill, or database delete. Translate its `ProjectManagerProtectedError` to the
   manager's public `InstanceError` so the API returns a deliberate client error.
5. Do not add agent startup, background retry, GPU probing, or a separate compute
   control plane to the default terminal lifecycle.

Production Nexus currently allows `/home/aday`; test/standalone configurations
that do not allow that exact workspace must fail resume validation instead of
silently changing the Project Manager's root.
