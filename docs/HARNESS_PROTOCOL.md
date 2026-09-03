# Aeon harness protocol

Aeon treats the language model as a proposal generator, not as the authority on
what happened. `aeon/core/agent_protocol.py` classifies the exact role=user
request, authorizes effects, normalizes tool receipts, tracks execution state,
and decides whether a final claim has enough evidence.

## One turn, one meaning

Every model decision has one of four kinds:

- `tool_calls`: request one mutation or a batch of independent reads; `message`
  is empty.
- `final`: publish one terminal response; no actions.
- `ask_user`: ask one necessary question and yield; no actions.
- `wait`: publish a real external/compute waiting condition and yield; no actions.

These combinations are constrained in the decode-time JSON schema as well as
validated again by the executor. Tool parameters come from each tool's exact
Python signature plus any narrower tool-defined required/cross-field contract;
unknown names, missing required fields, invalid edit forms, unknown fields, and
wrong JSON types fail before execution.

The user request remains an exact `role=user` message. Tool invocations are
stored as assistant `tool_calls`, and their normalized observations are stored as
matching `role=tool` receipts. Harness-owned policy metadata is trusted system
text; volatile project, file, memory, receipt, and job state is untrusted input
represented by a synthetic assistant call and matching tool receipt, never
upgraded to user or system authority. History trimming keeps each call and all
of its receipts atomic, retains only a bounded durable suffix, and chains a
deterministic digest over omitted groups. Hidden model reasoning is transient by
default.

Continuous mode is an owner-controlled scheduler around this turn boundary, not a
new turn kind. A mode-600 per-instance file supplies a validated goal of at least
three words. After any natural yield, the CLI may create a fresh autonomous request
whose prompt preserves all existing permissions and safety rules. The authenticated
stored goal alone derives the fresh request's mode, typed capability families, and
bound targets; scheduler prose and failed-cycle recovery evidence contribute zero
authority. It first clears a
`waiting_user` contract so the continuation can never masquerade as the user's
answer. Actual queued user text has priority. Disabling sends the private turn-stop
control to the exact live Aeon as well as persisting the disabled state, while
enabling an idle live process uses a separate private non-model wake signal.
Legacy interrupted continuous contracts that predate the separate authority field
resume read-only for that cycle rather than trusting wrapper-derived permissions;
the following fresh cycle rebinds from the owner-private control file.
The fresh request keeps the durable campaign's task/project memories, current plan,
recent history, and no-progress guards; only request-local action receipts are
cleared. A typed failed or blocked cycle is fed back as bounded harness-owned
recovery context, with cancellable 2/5/15/30/60-second backoff for equivalent
failures. The next cycle must change source, method, hypothesis, modality, scale, or
contribution instead of retrying an unchanged obstacle. A cycle final is therefore
a checkpoint, not evidence that continuous demand is exhausted or permission to
turn the mode off. Generation-budget failures contain no executed action or task
evidence, so their exact synthetic continuous-prompt/failure pairs are removed
from the next model projection instead of accumulating as a self-reinforcing loop;
the terminal harness message itself is not replayed into model history, and its
dangling synthetic prompt is discarded at the failure boundary. Legacy pairs are
also removed while restoring a checkpoint, before any new model projection can
see them. After three
semantically identical failed or blocked cycles, the CLI opens a circuit and yields
at its ordinary prompt rather than generating a fourth unchanged retry. Six
consecutive failed/blocked cycles also open the circuit, so alternating error text
cannot evade the liveness bound. The saved
continuous setting and goal remain enabled, so a user message, managed restart,
configuration correction, or mode toggle can resume from durable context. The
owner-visible transcript and all successful assistant/tool groups are retained.

Continuous research also preserves the ordinary evidence boundary. Retained
receipts are historical evidence, not authorization or automatic current
validation; freshness-sensitive claims must be checked again. A successful
write, reread, or digest proves only the bytes on disk; it does not validate claims
inside them. A zero-result search is never proof of absence. Candidate promotion is
prompted to wait for exact identity/version, release date, size and architecture,
license and derivative-redistribution review, real resource/toolchain requirements,
competing derivatives, differentiation, user value, and a reproducible validation
plan. When one lane is blocked, the scheduler continues the durable goal through a
different safe lane rather than reprinting the blocker.

For Hugging Face scouting campaigns this is also an executor gate, not prompt-only
advice. Strong candidate labels require current-cycle typed Hub receipts for one
exact repository, its revision/timestamps, architecture and numeric parameter
metadata, same-revision license text, and bounded competition sampling. Even those
receipts establish only the named facts: the harness still rejects winner,
decision-ready, confirmed-absence, and ecosystem-closure claims without the
separate feasibility, value, benchmark, or exhaustive evidence they require.
Provisional leads, explicit unknowns, and finite query results remain reportable.
The durable campaign ledger remembers explored branches and rejected strategies,
but prior-cycle receipts never satisfy a fresh factual gate.

## Persistent skill learning

Aeon ships only four broad, read-only base playbooks: codebase analysis, root-cause
debugging, change-propagation verification, and deep research. Learned skills live
in an owner-private overlay belonging to one durable agent (bounded to 16); no
runtime CRUD path can modify the packaged catalog or another agent's overlay.

The sibling skill wiki is the agent's editable offline context. The
`remember_skill_knowledge`, `list_skill_knowledge`, `search_skill_knowledge`,
`read_skill_knowledge`, and `delete_skill_knowledge` tools maintain bounded,
revision-checked notes. Ordinary first-try findings may be recorded there. A note
becomes skill evidence only when the harness receipts for that exact request show
at least one failed approach followed by a success, while the agent declares a
stable procedure, explicit verification, and low uncertainty. `create_skill`
requires one or more exact eligible note revisions and all learned playbooks must
state when to use them, their preconditions, procedure, verification, and when to
stop or adapt. One recovery episode can earn at most one skill; additional findings
remain wiki knowledge. Revisions require both the current skill digest and evidence
from a new recovery request. A revision-checked note update can retract an incorrect
learning claim, immediately making any citing skill need review.

Skills are untrusted prior experience, never system authority. Activation is
explicit and bound to one request, source scope, and content revision. A new
request, dashboard edit, origin change, missing metadata, or stale/quarantined
lifecycle unpins the playbook instead of silently adopting it. A failed live tool
result pauses the playbook and quarantines an agent-authored version; a blocked
result pauses it pending an honest `deactivate_skill` outcome. Successful,
adapted, failed, and not-applicable outcomes update revision-bound counters;
adaptation requires review. `delete_skill` is revision-safe retirement: it first
archives the old procedure and reason in the wiki. Notes and skills remain
editable evidence, must never contain credentials, and grant no tool or request
authority.

Skill text and wiki contents are loaded only through typed tool/live-state
observations, never copied into a `role=system` message. Storage is byte-bounded,
owner-only, no-symlink, and atomic. Each bounded sub-agent receives its own
overlay rather than inheriting the principal's learned state.

Authenticated Nexus users can copy selected private skills between ordinary
durable local Aeon tabs. The transfer binds each selection to its source revision,
never overwrites a different target version, and can copy related wiki notes with
source instance/note/revision provenance. A transferred procedure is intentionally
`needs_review` at the target and transferred notes do not count as locally earned
evidence. Shared catalog skills are already known and are not copied. Temporary
forks and public collaborator agents cannot transfer durable skills.

For a genuinely multi-step request, `updated_plan` is a concise user-visible
Markdown checklist rather than a hidden thought summary. The harness separately
compiles conservative owner-authored acceptance leaves (`G1`, `G2`, ...); the model
cannot satisfy them by checking off its own plan. `goal_refs` remain available as
optional precision hints, but the normal path binds typed receipts to matching leaf
targets after execution so the model does not have to operate internal bookkeeping.
Explicit aggregate refs are rejected, inferred evidence cannot cross unrelated
leaves, and every leaf retains its own typed mutation, inspection, validation, or
invariant receipts. A later relevant mutation invalidates old positive validation,
and an invariant observes every mutation even when the model omits that invariant's
ID. If an unusually large request exceeds the bounded goal compiler, completion
fails explicitly instead of silently dropping criteria.
Directive framing such as “your goal is to,” “you must,” “make sure to,” and a
bounded conditional preamble does not hide the action that follows it. Explicit
selection limits and parenthesized prohibitions become invariant leaves, while
research, build, publication, and documentation clauses retain separate evidence.

Aeon publishes each material checklist revision as a bounded, redacted `plan`
transcript record; an empty record clears a previous request's checklist. Nexus can
therefore pin the latest checklist below the append-only live progress stream
without exposing reasoning, parameters, commands, prompts, or raw tool output.

## Authority and evidence

Request modes are `answer`, `inspect`, `plan`, `change_local`,
`external_action`, and `destructive`. A polite direct request such as “Can you
refactor this file?” is actionable; a hypothetical question about how or whether
something could be done remains informational. An explicit later confirmation can
elevate an exact visible proposal. Negated actions do not become permissions.

Every concrete action is checked again at execution time. Dynamic shell commands
are classified from their actual command, and read-only versus mutable sub-agent
spawns are distinguished from their parameters. A newly queued user message
preempts a stale decision before it can publish or mutate state.

Consequential authority is also split into exact capability families and bound
targets. A GitHub push, agent registration, publication, expert consultation,
service/process control, source rollback, or deletion cannot stand in for another
family merely because both are mutations. A terse confirmation inherits only the
exact visible pending proposal; a path-only reply can bind an already-authorized
family but cannot introduce a new one.

Generic external effects are additionally bound to both operation and destination:
for example `send` plus a recipient, `post` plus a platform, or `publish` plus a
site. Browser/MCP calls must repeat those typed scopes and cannot switch recipient,
account, platform, site, or operation while recovering. A targetless request yields
only for the concrete missing scope; “Should I proceed?” and other generic
permission questions are rejected when the owner's request already grants the
necessary authority.
Quoted copy is excluded from authority classification, and descriptive modifiers
such as “authenticated,” “credentialed,” or “current” are never treated as literal
recipients or account names. Typed provider tools contribute their immutable
platform as observed scope, and platform-specific operation synonyms are
canonicalized only through reviewed metadata; they are not global aliases.

A mutation is an observation boundary: later actions and success prose are
deferred to a new model decision. Mutation requests need a successful change
receipt and later validation, or an explicit no-change/already-satisfied finding
grounded in a current receipt. Failed, blocked, pending, and unverified actions
cannot be restated as success. Inspection requests require current read-only
evidence unless they truthfully report a blocker.

File mutation receipts carry their target identity. Opening an unrelated file
does not satisfy post-edit validation; the changed target must be re-read or a
broader test/health probe must pass. `open_file` emits an explicit typed status:
arbitrary source text such as an error example, denial message, or running-state
fixture remains observed content and cannot reclassify a successful read as a
failure, policy block, or pending operation.

A sub-agent report is information, never validation of the principal workspace.
Even a successful report cannot clear validation debt opened by a local mutation;
the integrated principal state still needs its own targeted readback or validator.

`run_command` likewise treats its harness-generated first-line envelope as the
authoritative execution outcome. Captured stdout remains intact, but words such as
`Error:`, `permission denied`, `status: running`, or `NO CHANGE` inside successful
program output cannot override a verified `COMMAND SUCCESS` receipt. Failure,
timeout, block, and refusal envelopes retain their existing unsuccessful status;
command side-effect and change classification are unchanged.

Oversized tool output is evidence, not permanent prompt content. Above the
1,600-character inline boundary, Aeon keeps a bounded head/tail receipt and writes
the complete rendered result to the instance's owner-private state under an opaque,
request-scoped reference with SHA-256 integrity. `inspect_tool_result` accepts only
that reference—not a path—and provides literal search or a page of at most 3,000
characters. All inspections in one model turn share an 8,000-character budget,
exact repeats return a stub, only eight recent references remain in live harness
state, and a new request cannot address the previous request's files. Storage is
append-only and hard-quota bounded (8 MiB/result, 16 MiB/request, 64 MiB/instance);
quota or archive failure never changes the original tool status and never triggers
automatic deletion.

The model loop is finite independently of model compliance. One request gets at
most 64 decision turns. One decision and all of its recovery/search calls share a
six-call, 65,536-completion-token, 1,800-second generation budget; if candidate search
has already produced a valid proposal, later budget exhaustion retains that
proposal instead of wasting the decision. If the attempt still exhausts that
budget before a usable turn, the worker permits one separate compact recovery:
candidate search and hidden thinking are disabled, its prompt reserves only the
smaller output allowance, and one low-effort semantic generation shares an
8,192-output-token aggregate, six-call compatibility-negotiation, 180-second wall
budget. A proposal-specific length exhaustion does not abort a still-affordable
independent candidate; if none succeeds, the typed generation-budget failure is
preserved rather than being converted to a generic retryable error. The support-
model path has its own smaller shared ceiling.

No-progress handling is event-triggered rather than a fixed retry counter. The
first grounded failure bars the exact call and requests diagnosis; recurrence
requires a different mechanism; alternating A/B failures require a parent-goal
reframe and an independent third method. Rewording or parameter churn does not
count as strategy diversity. A hard stop occurs only after the model repeatedly
ignores an exact bar or a bounded multi-strategy plateau is proven. Only an explicit
evidence/acceptance advance resets the recovery epoch. Routine execution uses one
medium-effort proposal, a complex first decomposition uses high effort, and recovery
selectively receives two or three independently generated candidates. This keeps
the fast path short without asking the same hypothesis to “think harder.”

A bounded, non-memory session strategy ledger survives context projection and
checkpoint restore. It retains factual method/outcome/goal events—not hidden
chain-of-thought—so compaction does not erase rejected routes, open criteria, or
the basis for a reframe. Exact read-only transient failures receive at most one
cancellable same-call retry; permanent failures, mutations, queued user input, and
stop signals never enter that retry path. Terminal blocker prose is accepted only
after a latest typed non-retryable invariant receipt. `wait` is accepted for
compute only when the latest observation is a verified active durable Fleet
receipt; model prose or a port alone cannot create waiting state.

For public Hugging Face research, three fixed read-only tools expose paginated model
search, exact repository metadata, and bounded repository files at a named revision.
They use the public HTTPS origin without inherited credentials and return source
data rather than an LLM summary. Their receipts explicitly retain the distinction
between metadata/card claims and verified facts, and between a negative query and
proof that no competing repository exists.

Nexus first-class provider credentials and remote MCP account connections are
separate inventories. `list_provider_credentials` returns only metadata for exact
provider grants, including Hugging Face; it never returns a secret and never treats
the grant itself as an upload capability. `list_mcp_credentials` therefore must not
be polled for a site credential that cannot appear there.

## Safety boundaries

- Existing-file edits are bound to the SHA-256 returned by `open_file`; stale,
  ambiguous, fuzzy-by-default, symlink, blind-overwrite, and syntactically invalid
  edits fail before replacement.
- Direct file tools are confined to the immutable launch workspace using
  descriptor-relative no-follow traversal. Sensitive state/credential roots,
  symlink ancestors, multiply-linked files, and launch-root replacement are
  refused. Writes recheck exact inode/version identity and replace atomically.
- Memory is scoped (`task`, `project`, or `preference`), can expire, and rejects
  credential-like names or values. Legacy secret-like entries are withheld.
- Read-only sub-agents share the workspace under an enforced inspect contract.
  Mutable sub-agents require a clean Git snapshot and an isolated detached
  worktree, and the wrapper enters that worktree before constructing any worker or
  tool. At terminal state the wrapper uses a private temporary Git index to freeze
  an uncommitted, regular-file-only binary patch with exact base, path, mode, size,
  and SHA-256 receipts. Report previews do not count as collection; reports are
  paged from an explicit offset and become collected only when the caller reaches
  EOF. Integration then requires that exact changeset ID, proven child-process
  absence, unchanged parent HEAD and affected paths, sensitive/protected/invariant
  admission, a no-write conflict check, and exact post-apply hashes. A PREPARED /
  APPLIED journal prevents crash replay. Unrelated dirty principal files are
  preserved, Git refs/index are untouched, and post-integration validation remains
  mandatory.
- Session, request, blackboard, and sub-agent state live in owner-private storage
  outside the source tree. Waiting request contracts and typed receipts survive a
  state round trip. Session checkpoints have a symmetric 8 MiB ceiling, stable
  owner/nofollow reads, and atomic file-plus-directory fsync writes.
- External expert consultation is default-off and requires both explicit external
  authority and operator opt-in.
- Every registered tool has one explicit compute route. Registration and the
  immediate call boundary reject missing or changed declarations. Local-model
  helper calls revalidate the current Fleet ticket; Comfy media calls and local
  sub-agents own separate exact tickets and loopback endpoints.
- Generic synchronous and background shell calls are CPU-only. Their concrete
  command is admitted before any process, heartbeat thread, or durable job state
  exists; direct GPU/coordinator/device/lease and recognized scope-escape paths
  fail closed with instructions to use a reviewed Fleet profile. Each admitted
  command uses a unique gated transient user service; exact systemd identity and
  hardened-property readback plus a durable InvocationID receipt precede gate
  release. Its scrubbed environment uses exact `CUDA_VISIBLE_DEVICES=void` for
  coordinator-recognized CPU-only CUDA intent while leaving the independent
  ordinal and AMD no-device sentinels at `-1`; closed device policy and Landlock
  remain the enforcement boundary. Landlock actively proves guardrail
  write/create/rename/unlink denial,
  credential/coordinator read denial, and non-standard-device denial inside the
  actual unit. Seccomp actively proves that AF_UNIX, IPv4, IPv6, and netlink socket
  creation all fail. Generic commands therefore have no network access; source
  writes use the receipt-bound file tools, while a private cwd-local scratch path
  supports temporary command output. The command controller and transient bootstrap
  are bound to their launch-time source digests; an edited deployment is a typed,
  non-transient restart requirement rather than an invitation to retry an
  incompatible old parent against new child code.
- On the current host, the unprivileged user manager accepts and reports
  `DevicePolicy=closed`, `ProtectSystem`, path mounts, and IP ACL properties but
  does not enforce their device/path/IP effects. They remain required, externally
  read-back defense in depth, never the proof of containment. The in-unit Landlock
  and seccomp probes above are the enforcement authority; a missing Landlock ABI,
  unreadable harmless device baseline, or failed probe leaves the gate closed.
- Per-turn system context does not poll the coordinator or Fleet broker. Compute
  state comes from the owning durable job/runtime path, avoiding hidden control-
  plane latency and respecting Fleet's single-broker boundary.
- Parallel execution is restricted to batches of two or more allowlisted,
  stateless read tools, with at most four workers. Shell, mutation, Fleet/runtime,
  active-model, Nexus-lifecycle, and external-provider actions remain serialized.
- Nexus chat delivery uses a stable client turn identity and an exact
  claim→prepare→private-paste→transcript-commit transaction. Retries accept only
  the same visible text, attachment metadata and attachment SHA-256; ambiguous
  post-paste failures never trigger a second paste. Receiver waits, corrupt-state
  handling, and lock acquisition are all bounded.

## Verification

The fast model-free gate is `./run_tests.sh`. The protocol-specific regressions
are in `test_agent_protocol.py`, `test_worker_protocol.py`, and
`test_harness_safety.py`; they use scripted models and fake tools and do not
require GPU compute or live services.
