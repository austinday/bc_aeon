# Aeon capability benchmark

Nexus exposes one benchmark: the complete, fixed agent-capability benchmark.
There is no user-selectable suite. A run compares one reviewed combination of
harness, logical model service, and tool profile; **All** expands only the
explicit combinations in the immutable catalog, never a Cartesian product.

The current reviewed combinations are:

| Harness | Logical model service | Tool profile |
|---|---|---|
| OpenCode `1.18.27` | `local/qwen` | `fleet-local` |
| Legacy Aeon `0.2.0` | `local/qwen` | `fleet-local` |

`local/qwen` means the reviewed logical Fleet service
`aeon-qwen38-standard`. It does not assert that a particular concrete model
artifact served a run. Fleet may route among compatible READY lanes according
to owner policy and safe availability. Results compare the recorded logical
service identities unless future transport evidence truthfully identifies a
more specific backend.

Benchmark workers start through `/home/aday/bin/fleet-low-priority`. Harnesses
obtain GPU-backed model service only through Fleet Compute. The benchmark never
selects a machine, GPU, claim, device, endpoint, or renter resource, and never
creates a second allocator or warm reservation. Broker-proven unavailability is
durable waiting rather than a model failure.

## What the benchmark measures

The fixed benchmark evaluates agent behavior, not the quality of the tools it
calls:

- instruction following and uncertainty: exact multi-constraint output,
  no-tool requests, safe clarification before mutation, and truthful unknowns;
- memory and context: implicit facts, later supersession, browser-session state,
  and transformation after deterministic low/medium/high context pressure;
- tool-call judgment: whether a tool is needed, exact reviewed-tool selection,
  bounded arguments and call count, purpose-built file tools instead of ad hoc
  shell/code, and stopping after an unchanged failed action;
- web and visual reasoning: controlled reading, form completion, session use,
  screenshot grounding, and direct image inspection;
- Fleet resilience: appropriate durable compute demand, useful CPU work while
  queued, preemption recognition, checkpoint continuity, reacquisition, and
  same-job resume without duplicate submission or spin;
- parallel orchestration: independent delegation, useful principal overlap,
  dependency gating, bounded concurrency, result collection, and verified
  integration; and
- reliability and whole-task efficiency: verified completion without stalls,
  scored by total active time across the task rather than the duration of each
  individual reasoning step.

No live email account, credential, CAPTCHA, arbitrary public website, or renter
workload is used by the benchmark. Browser and visual cases use immutable,
authenticated local fixtures. Fleet and parallel cases use hidden deterministic
fake state machines and effect ledgers; they test decisions and recovery
semantics without submitting real GPU work or spawning real delegated jobs.
Fixture or harness readiness failure invalidates the run and publishes no
comparable score. It is never converted into a model zero.

The context-pressure case introduces facts as ordinary project context, then
adds seven resumed 32,000-byte turns of deterministic, SHA-256-derived
entropy-dense material. It checks recall after 32,000, 96,000, and 224,000
cumulative bytes while every individual authority request remains below the
OpenCode 40KB hard limit. The raw stimulus is sufficient to exceed the nominal
context-minus-output threshold if it were retained in full; actual occupancy is
reported only from provider usage rather than inferred from cumulative bytes.
After the final recall it requires one exact purpose-built
file-tool call, with no shell or ad hoc code. It reports
`context_pressure_bytes`, `context_pressure_turns`,
`highest_verified_context_pressure_bytes`, and the authoritative
`peak_prompt_tokens` observed from provider receipts when that evidence is
complete. This is retention and tool judgment under controlled pressure; it is
not labeled as a proven compaction event. Neither compaction nor threshold
crossing is inferred when usage evidence is incomplete.

## Scores and raw measurements

Each behavioral case is scored from independently verified output, effects, or
authenticated typed receipts. Component case weights have fixed denominators;
missing, timed-out, or unexecuted cases are not dropped or renormalized.

The component weights are:

| Component | Weight |
|---|---:|
| Instruction following and uncertainty | 20% |
| Memory and context | 20% |
| Tool-call judgment | 20% |
| Web and visual reasoning | 10% |
| Fleet resilience | 10% |
| Parallel orchestration | 10% |
| Reliability and whole-task efficiency | 10% |

`overall_score` is the weighted score on a 0–100 scale. The legacy `score`
field remains as `overall_score / 100` for old readers. `quality_score` is the
weighted mean of the six behavioral components excluding
reliability/efficiency. Nexus uses behavioral quality versus
`total_active_wall_ms` for Pareto comparison so time is not counted twice on
both axes.

The reliability/efficiency component is half verified completion and half
whole-run efficiency. Let `D` be the sum of fixed case deadlines, `T` the sum of
fixed active-time targets, and `A` the sum of active wall time charged to the
run. A missing, timed-out, or stuck case consumes its full deadline. Then:

```text
efficiency = clamp((D - A) / (D - T), 0, 1)
```

A run at or below its total targets receives 100 efficiency; a run at or beyond
its total deadlines receives zero. One long reasoning call can therefore beat
many short calls when it reaches verified completion sooner. Fleet queue time
is reported separately and excluded only when the broker proves the wait.

Summaries expose:

- `total_wall_ms`, `total_active_wall_ms`, and `total_compute_wait_ms`;
- `model_turn_count`, `model_call_count`, and `tool_call_count`;
- cumulative provider-reported input (`prompt_tokens`), output
  (`completion_tokens`), and total (`context_tokens`) tokens, plus the maximum
  input size of any individual call (`peak_prompt_tokens`) and
  `token_metrics_complete`; `context_tokens` is total consumed across calls,
  while `peak_prompt_tokens` is the measured high-water mark;
- controlled context-pressure byte/turn coverage;
- Fleet decision, recovery, useful-wait-work, checkpoint/reacquire, and
  duplicate-submission evidence; and
- parallel useful-overlap, idle-wait, maximum-concurrency, and integration
  evidence.

`model_call_count` covers every task-attributable local generation transport:
OpenCode reasoning/compaction requests through its Fleet proxy and legacy Aeon
decisions, retries, verification, skill routing, memory/action-log compaction,
resume integration, and model-backed support tools. Readiness/health probes and
separately delegated sub-agent transports are excluded. A durable start receipt
is required before the network call can proceed, so failed retries still count.
Token totals appear only when every started call finishes successfully with a
complete, internally consistent provider usage object. One failed, interrupted,
missing-usage, or partial-usage call makes all case token fields `null`; counts
remain exact when the start evidence is intact. Values are never estimated.
Total active time spans every turn of a multi-turn case.

## Tool and simulator evidence

Model prose and recognizable tool-output text are not evidence of a call.
Executor-created, owner-private capabilities bind receipts to the exact run,
case, repetition, random nonce, file identity, monotonic sequence, operation,
status, and bounded effect facts. Raw sensitive arguments are not retained.
Malformed, missing, duplicate, reordered, replayed, cross-case, or tampered
streams fail closed.

The hidden Fleet fixture has stable opaque job and checkpoint generations. Its
driver controls queued and preempted transitions, independently verifies useful
CPU work within the wait window, and requires job/checkpoint continuity on
resume. The hidden parallel fixture models independent branches, principal-only
work, a digest-gated dependent branch, bounded delegation, and final integration.
Its virtual intervals support overlap and idle-wait measurements without timing
flaky real subprocesses. Merely reciting the expected action order produces no
effects and cannot pass.

All `AEON_BENCHMARK_*` variables are principal-only. Model-invoked commands and
subagents have them scrubbed, and scenario capabilities exist only for the
applicable case. Evidence files live outside the model-facing workspace.

## Durable runs, matrices, and comparison

An exact run submission may omit `suite_id`; the service always binds the
canonical internal ID `comprehensive`:

```json
{
  "request_id": "br-00000000000000000000000000000000",
  "harness_id": "opencode",
  "model_id": "local/qwen",
  "tool_profile_id": "fleet-local",
  "repetitions": 1
}
```

`request_id` is a durable idempotency key. An identical retry returns the same
run, while a changed body is rejected. Historical rows for old partial suites
remain readable, and an exact lost-response retry of such a row still works;
new partial-suite submissions are rejected.

The matrix primitive accepts `all` independently for harness, model, and tool
profile while selecting only catalog-reviewed, currently runnable rows:

```json
{
  "request_id": "bm-00000000000000000000000000000000",
  "harness_id": "all",
  "model_id": "all",
  "tool_profile_id": "all",
  "repetitions": 1,
  "missing_only": true
}
```

`missing_only: true` is the prominent **Run all missing** operation. Within one
transaction the server reuses an active current run or an evidence-verified
succeeded run and creates a durable `pending` child only for missing coverage.
Currentness binds the exact catalog, suite, runner/executor protocols, source
digests, combination identity, and repetition count. Corrupt evidence or stale
provenance is missing. Concurrent tabs cannot create duplicate missing work.

A committed batch retains its original child mapping across retries and later
catalog drift. At most one matrix-created child is active globally; terminal
completion, cancellation, stale-worker reconciliation, and service startup all
advance the next durable child. `cancel_batch` cancels only children created by
that batch and preserves finished or reused runs. Explicit single-run submissions
remain independent.

`comparison(repetitions=1)` is the server-authoritative coverage view. It returns
one entry per reviewed combination with `state` (`succeeded`, `active`, `failed`,
or `missing`), `submission_available`, `needs_run`, `evidence_verified`, and the
sanitized current run when one exists. The UI must not infer missing coverage
from a truncated newest-runs list.

The intended authenticated Nexus routes are:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/benchmarks/catalog` | Canonical benchmark, scoring metadata, combinations, readiness |
| `GET` | `/api/benchmarks/comparison?repetitions=1` | Current comparison and missing coverage |
| `GET` | `/api/benchmarks/runs?limit=100` | Newest historical summaries |
| `GET` | `/api/benchmarks/runs/{run_id}` | Hash-verified case evidence |
| `POST` | `/api/benchmarks/runs` | Submit one exact combination |
| `POST` | `/api/benchmarks/runs/{run_id}/cancel` | Cancel one run |
| `POST` | `/api/benchmarks/matrices` | Submit an idempotent exact/all matrix |
| `GET` | `/api/benchmarks/matrices/{batch_id}` | Read durable batch progress |
| `DELETE` | `/api/benchmarks/matrices/{batch_id}` | Cancel batch-owned unfinished work |

Every route requires a Nexus session; writes additionally use normal CSRF,
Origin, and audit controls. Browser responses contain no raw prompt, credential,
endpoint, command, model path, Fleet ticket, claim, UUID, device, PID, tool
output, simulator secret, or raw tool arguments.

## State, provenance, and cancellation

The CLI state root defaults to `~/.local/share/aeon/benchmarks`, or an absolute
owner-private path supplied through `AEON_BENCHMARK_HOME` or `--root`:

```bash
cd /home/aday/NexusAgentDashboard/bc_aeon
python3 -m aeon.benchmarks catalog
python3 -m aeon.benchmarks list --limit 20
python3 -m aeon.benchmarks show run-0123456789abcdef0123456789abcdef
python3 -m aeon.benchmarks cancel run-0123456789abcdef0123456789abcdef
python3 -m aeon.benchmarks submit \
  --harness opencode \
  --model local/qwen \
  --tool-profile fleet-local \
  --repetitions 1
```

Nexus uses its own owner-private `state_dir/benchmarks`. Directories are mode
`0700`; databases and evidence are singly linked owner files with mode `0600`.
SQLite queue state and bounded JSON evidence remain separate from source and
from model-facing workspaces. Readback verifies evidence ownership, size, shape,
run identity, and SHA-256 before returning cases. Failed verification returns no
untrusted cases.

Workers bind their exact owner PID and kernel start time in the private database.
Reconciliation marks a vanished exact worker `failed/worker_lost`; an
unregistered queued worker receives a bounded startup grace. Process identity is
never exposed through Nexus. Cancellation targets only the proven run-owned
process tree and revalidates PIDs before signaling. It never searches or kills by
process name.

Provenance binds catalog and canonical benchmark versions, request,
runner/executor protocols, exact combination, evidence digest, and content-bound
runner, harness, tool, prompt/scoring, fixture, and simulator sources. A queued
run refuses to execute after source drift. Compare results only when those
identities match; `comparison` and **Run all missing** enforce this server-side.

## Extending the benchmark

1. Add only reviewed harness/model/tool identities and explicit
   `CombinationSpec` rows. Never generate combinations from independent lists.
2. Add a semantic `ScenarioSpec`; keep prompts, paths, fixture state, secrets,
   and expected effects out of the public catalog.
3. Verify agent behavior independently. Tool prose is insufficient, and fixture
   quality must not contribute to model scoring.
4. Use a case-scoped hidden capability and deterministic FSM/DAG for simulated
   external systems. Test replay, ordering, context binding, file identity,
   effect continuity, duplicates, and denied/error operations.
5. Give every dependency a fail-closed preflight. Infrastructure failure must
   invalidate the run rather than score zero.
6. Set fixed target/deadline bounds; bump benchmark/catalog/protocol/source
   provenance for any semantic change.
7. Add fake-backed scoring, timeout, cancellation, restart, matrix-idempotency,
   evidence-tamper, environment-scrub, and migration tests.

## Verification

The benchmark regressions are fake-backed and do not acquire live GPU compute:

```bash
cd /home/aday/NexusAgentDashboard/bc_aeon
python3 -m pytest -q \
  aeon/tests/test_benchmarks.py \
  aeon/tests/test_benchmark_scoring.py \
  aeon/tests/test_benchmark_matrix.py
```

Run the matching Nexus API and browser tests from `dashboard` for cross-project
changes. Deployment, service restart, or a live benchmark requires separate
operator authority and the Fleet operations runbook; this test workflow does
not perform them.
