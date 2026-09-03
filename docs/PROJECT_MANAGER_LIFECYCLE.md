# Nexus main orchestrator lifecycle

The main orchestrator is an always-present, renameable Nexus chat identity, not an
always-running agent. Its durable registry row is rooted at `/home/aday` on `DAY2RTX6000PRO`
and is initially created with `status=idle` and `desired_state=stopped`. Nexus
application assembly opens only its fixed Bash tmux shell. This base-shell start
does not start Aeon, a provider CLI, a model runtime, or a GPU coordinator lease.

`aeon.remote.project_manager` owns the stable UUID-shaped ID and protected
identity fields. `ensure_project_manager()` performs only registry `get`/`create`
operations. It accepts a concurrent creator only after re-reading and validating
the complete protected identity. A mismatch fails closed rather than adopting or
rewriting an uncertain row.

## Activation boundary

Only Nexus application assembly may open the virgin fixed shell automatically.
Ordinary list/status reads remain side-effect free. Entering Agents makes the
explicit authenticated, CSRF-protected, idempotent start request; there is no
background relaunch loop. The manager verifies the exact base shell, migrates an
old dormant workspace to `/home/aday`, and starts only the fixed Aeon command.

Aeon receives the fixed main-orchestrator instruction through a private marker,
not through a fabricated startup user message. It therefore arrives at the prompt
ready for the user's first real message. Ending it returns the same durable instance
to Bash; its ID, name, transcript, prompt versions, and local-role versions remain.
The authenticated **Restart Nexus** control instead makes one fresh-context start
request. Under the same manager lifecycle lock, it ends any verified foreground,
atomically replaces that instance's durable Worker checkpoint with empty state,
removes its interrupted-resume checkpoint, clears its private chat transcript, and
only then activates a new Aeon process. The durable tab, name, identity, instruction
layers, settings, and capabilities remain. A reset failure leaves a durable
fresh-context-required guard that blocks ordinary supervision from relaunching old
memory until a later verified reset succeeds. The browser never attaches to the
protected shell or replaces the durable registry row.

## Chat boundary

The browser never attaches to the underlying PTY. Nexus reads a bounded,
owner-private `chat-transcript.jsonl` containing only `user` and `assistant`
messages. A user message is delivered only when the exact protected instance has a
verified managed Aeon foreground; delivery uses a private tmux buffer and bracketed
paste, and the transcript is appended only after tmux accepts it. Assistant replies
are written best-effort by Aeon's top-level process. The transcript environment also
contains that exact writer PID, so inherited subagents cannot impersonate the main
orchestrator. Transcript paths reject symlinks, unexpected owners, extra links,
oversized files, malformed events, and control characters.

Messages submitted while an objective is active enter Aeon's process-local FIFO
and do not interrupt or rewrite that objective. One visible `say_to_user` response
ends the autonomous turn unless it immediately enters `get_user_input` or
`task_complete`. The authenticated Stop action delivers a private turn-control
line to the exact managed Aeon foreground. It may asynchronously interrupt only
the active model-generation scope. During a tool it records a cooperative stop:
the current tool finishes and its receipt is committed, then the Worker cancels
before another decision or action. A tool blocked on solicited console input is
unblocked with a bounded stopped receipt. The control never escapes as a process-
level interrupt, preserves the session and FIFO, and is never a transcript message;
the next queued user message starts automatically.

Only that Project Manager Aeon receives Nexus's owner-only local capability for
`start_agent_instance`. The tool registers a separate durable Nexus session in an
existing `DAY2RTX6000PRO` directory; it is not a nested Aeon sub-agent. A newly registered
Aeon has `status=idle`, `desired_state=stopped`, and an explicit persisted
`awaiting_objective` state. Registration does not create a tmux process, submit
model demand, analyze the workspace, or invent an objective. Nexus renders the
session as another linked, independently manageable Agents tab. Ordinary
Aeon instances and bounded sub-agents do not receive this capability or inherit
the Project Manager identity and bearer environment. The bridge reuses the existing
instance manager and workspace validation and does not allocate GPU compute or
bypass Fleet Compute.

The bridge may bind the new tab to one exact active Nexus project ID. Nexus
revalidates that the agent workspace is the project's exact canonical root; when
the ID is omitted, the unique active project with that exact root is associated
automatically. The returned typed receipt carries the resulting project identity,
so a mismatched bridge response cannot be accepted as evidence for an explicitly
selected project.

An interrupted managed Aeon process is relaunched through Nexus's private
`--resume-unfinished` boundary. The worker restores only that exact instance's
owner-private checkpoint when its request contract is still `running`; completed,
blocked, waiting, failed, and user-cancelled turns are never replayed. Durable
agent-creation guard state and verified receipts are checkpointed with the request,
so a reload cannot forget a terse confirmation or duplicate an external creation.
An explicit Nexus End transition atomically changes any stale `running` worker
checkpoint to `cancelled` after the exact managed process returns to its base shell,
so a later lifecycle resume cannot revive work the owner deliberately ended.

The same owner-only local control boundary exposes `connect_mcp_account` to the
Project Manager. It resolves a reviewed service such as Gmail, returns the
provider's browser authorization URL, and lets Nexus persist the completed grant
as a named reusable connection. `start_agent_instance(allowed_credentials=[...])`
can set the new durable tab's exact ACL at creation; omission grants nothing.
Ordinary and bounded agents can list or call only connections allowed by their
ACL, and `spawn_sub_agent(allowed_credentials=[...])` mints an expiring proxy
capability limited to the intersection with its parent's ACL. OAuth refresh and
remote MCP traffic run in Nexus, so raw provider tokens never enter agent or child
process environments.

Public Bitcoin receiving addresses contain no secret or signing material. A
managed agent can read only addresses explicitly allowed for its durable tab with
`list_payment_addresses`, and may share one when the user asks for a donation or
payment line. Bounded sub-agents receive only the parent-approved intersection.
Nexus rejects private keys, extended private keys, and seed phrases; the capability
cannot sign transactions or spend funds.

The user's first message in that tab is durably bound to the deferred instance and
becomes its exact one-time `--start` objective. It is written once to the structured
chat transcript and is not also pasted into the new PTY. A retry with the same
message identity returns the same transcript event and cannot launch or deliver the
objective twice. Resume on a never-started deferred row remains idle; Resume is not
authorization to synthesize an empty objective.

Questions about whether or how Nexus could create an agent request an explanation
or plan only. A start requires explicit present-action authorization from the user.
This distinction is enforced by a deterministic Project Manager turn guard rather
than prompt compliance alone. The guard bypasses generic skill routing for Nexus
agent lifecycle requests and permits only planning/waiting for capability questions.
For an authorized creation request, it blocks shell, file, memory, Ollama, and
health-check actions until the Project Manager calls `start_agent_instance` or asks
for required input. After calling the bridge, the Project Manager must observe its
result on a later model turn before reporting success; it must never pre-compose a
success claim in the same action batch.

The typed request contract classifies that lifecycle mutation as an external Nexus
action so the bridge remains present in the model's constrained tool schema. Native
`ask_user` turns retain the authorized creation transaction across a directory/name
clarification (including a process restart), while an explicit cancellation clears
it. Confirmation and clarification replies are reclassified by the durable guard
before any tool schema is exposed, and only the bridge's typed receipt unlocks a
success response.

Only a typed receipt built from the authenticated bridge's validated durable
instance record proves registration. For a deferred Aeon the receipt says it is
idle and awaiting the user's first message; it is not evidence that work started.
A prior transcript claim, persisted memory, file,
README, process, port, model response, or success-looking string is not evidence and
cannot unlock a visible success report or `task_complete`. The guard state and its
receipt reset at each unrelated user turn and on `/clear`.

Aeon's Fleet-broker compute path remains authoritative after activation. No capacity
is reserved merely to keep the chat warm unless the authenticated Nexus
`aeon-qwen38-standard` replica setting explicitly establishes a durable reviewed
serving target. During foreground demand, Aeon presence exposes
`waiting_for_compute`; it polls only its opaque broker ticket while Fleet's
coordinator attempts use bounded backoff and hold no lease between failed attempts.
This is not a reason to call the coordinator directly or create an independent
allocator. An exact lifecycle stop cancels/releases that agent's demand, records
compute as `unavailable`, and exits Aeon while the durable instance remains
resumable; a separately configured shared serving target remains Fleet-owned. A
process, service, or machine interruption also leaves that registry row resumable.
A dead pane must never continue to look actively queued.

Stopping the instance releases its ordinary Aeon resources and returns the row to
`idle` after its pane exits. The row itself cannot be deleted. Force-stop may end
an exact live Project Manager tmux session, but it must also preserve the row and
return to `idle` on reconciliation.

Ordinary child tabs have three distinct lifecycle actions. Stop requests a
graceful process exit and preserves the resumable tab. Exact-name force-stop proves
the exact managed tmux session absent and also preserves the tab. Exact-name Kill
holds the same per-instance lifecycle lock across that verified force-stop and
durable row deletion, so a resume or rename cannot race between them. If process
absence or deletion cannot be proved, the action fails without reporting
`deleted=true`; any surviving row remains visible for recovery. The protected Main
orchestrator guard runs before any pane lookup or signal, so Kill can never remove
or incidentally stop its permanent row.

Owner-enabled continuous Aeons are reconciled by the same five-second Nexus
lifecycle supervisor as the Main orchestrator. Recovery is authorized only while
the durable row still requests `running`, continuous mode remains enabled, and the
row is neither awaiting an objective nor attached to a collaborator portal. A
managed-shell row may be reactivated only from its exact private base prompt. A
legacy direct row is eligible only when it has the canonical local Aeon identity;
every live pane, including an unexpected foreground, is left untouched. Only an
independently proven dead or absent pane is resumed from its unfinished checkpoint.
Failures retain owner intent and retry with bounded 5--300 second backoff. Stop,
End, continuous-mode disable, force-state ambiguity, and collaborator ownership
always win over supervision.

## Manager integration contract

1. Call `ensure_project_manager(store, default_model=...)` while constructing the
   instance manager, before mutation endpoints become reachable. This closes the
   name/identity race without launching anything. It is safe to re-ensure before
   returning the instance list. During Nexus application assembly only, launch
   the fixed Bash shell when the row is still virgin and idle. Model activation is
   reserved for the authenticated main-orchestrator start capability; never reserve
   compute while merely assembling or listing the application.
2. Merge `project_manager_public_flags(record)` into the public instance object.
3. Consult `dormant_project_manager_status()` before generic stopped-session
   reconciliation so an absent/dead pane plus `desired_state=stopped` is `idle`.
   A missing pane while desired running remains truthfully `interrupted`.
4. Call `reject_project_manager_deletion(instance_id)` before any pane lookup,
   kill, or database delete. Translate its `ProjectManagerProtectedError` to the
   manager's public `InstanceError` so the API returns a deliberate client error.
5. Do not add background retry, GPU probing, or a separate compute control plane to
   the base-shell lifecycle. The authenticated start may create only ordinary Fleet
   demand and must expose a truthful wait when capacity is unavailable.

Production Nexus currently allows `/home/aday`; test/standalone configurations
that do not allow that exact workspace must fail resume validation instead of
silently changing the Project Manager's root.
