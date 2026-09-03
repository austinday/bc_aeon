# Aeon harness benchmarks

The benchmark system compares reviewed Aeon harness, logical-model-service, and tool-profile
combinations without turning Nexus into another allocator. The durable worker is
started through `/home/aday/bin/fleet-low-priority`; each real harness then
obtains its model only through Fleet Compute. The runner never selects a host,
GPU, claim, device, endpoint, or renter resource.

The current reviewed combinations are:

| Harness | Logical model service | Tool profile |
|---|---|---|
| OpenCode `1.18.27` | `local/qwen` | `fleet-local` |
| Legacy Aeon `0.2.0` | `local/qwen` | `fleet-local` |

`local/qwen` is the reviewed logical Fleet service
`aeon-qwen38-standard`, not a claim that one concrete artifact handled the run.
Fleet currently qualifies Flash-Next and 27B fallback profiles behind that service
and may route among compatible READY lanes according to owner policy and safe
availability. The router does not currently attest the selected backend for each
request, so these results compare harnesses while holding the logical service
constant; they must not be presented as Flash-Next-versus-27B measurements.

An unavailable binary, package, wrapper, or broker is a truthful failure outcome.
When the harness's one durable Fleet demand is admitted but has no safe placement,
the run remains `waiting_for_compute` without spending its case-execution budget.
The runner never substitutes another harness or model and never manufactures a
passing result.

## Suites and metrics

The immutable catalog in `aeon/benchmarks/catalog.py` currently offers:

- `smoke`: direct and bounded completion;
- `tools`: local read, sandboxed mutation with verification, and Fleet-wait
  semantics;
- `browser`: controlled observation, form, and session workflows;
- `vision`: fixture-image and browser-screenshot grounding;
- `context`: early recall, compaction recovery, and unchanged-action loop
  avoidance;
- `comprehensive`: every case above.

Each public suite record includes a sorted `required_capabilities` array. A
submission is admitted only when that set is a subset of the selected tool
profile's public `capabilities`; an otherwise valid harness/model/tool
combination is rejected fail-closed when it cannot execute the selected suite.
Catalog clients should use the same subset rule when filtering tool-profile
choices, while keeping the server check authoritative.

Every case has its own deadline and returns one of `passed`, `failed`,
`timeout`, `stuck`, or `unsupported`. Each case records `wall_ms` (total
end-to-end time), `compute_wait_ms` (only broker-proven waiting), and
`active_wall_ms` (`wall_ms - compute_wait_ms`). Run summaries report all three
medians; harness/model comparisons use `median_active_wall_ms`, while the legacy
`median_wall_ms` remains the total-wall metric. Summaries also report mean score,
completion rate, stuck rate, unsupported rate, tool and browser success rates,
vision score, case count, and passed-case count.
`unsupported` is a measured zero, not a pass or omitted sample.

`tools.fleet_wait` cannot pass from an echoed marker or model explanation. The
reviewed `fleet_batch_capabilities` implementation emits one bounded,
HMAC-authenticated typed receipt into an executor-created owner-private file;
the executor independently requires its exact recipe-only and durable-wait
fields. Duplicate, malformed, missing, or tampered receipts fail the case.

Browser and `vision.browser` cases use the authenticated browser service's
non-model-exposed `/benchmark_fixture` endpoint. It accepts only immutable
server-owned fixture IDs plus a fixed benchmark profile and unique session; it
accepts no caller HTML, script, URL, or path and performs no network navigation.
The fixtures cover observation, a multi-field form, session continuity, and
screenshot grounding while ordinary browser navigation keeps its public-HTTP
SSRF checks. A stale browser-service image makes these cases truthfully
`unsupported` until the reviewed image is rebuilt and receipted. Remaining
limitations are explicit: `context.compaction` has no deterministic public
compaction trigger, and legacy Aeon cannot run the multi-process context-recall
case. No case claims that an arbitrary live site, email account, CAPTCHA, real
login, or production credential was tested.

The session fixture stores only a synthetic per-run marker in origin-scoped
browser storage. After the first harness turn signs in, the authenticated
fixture endpoint independently verifies that marker, closes the exact tab, and
opens a fresh tab in the same persistent context. A separate resumed harness
turn must continue the session before final verification. The fixed
`aeon-benchmark.invalid` document is fulfilled by an in-page route from compiled
fixture bytes; no request reaches that name and no real credential is used. The
executor then removes the exact per-run marker during normal fixture cleanup.

## Nexus page and API

The authenticated **Benchmarks** page loads the server-owned catalog, filters
models by harness, submits one exact combination, compares result summaries,
and lazily loads case evidence and provenance. It polls only while the page is
visible and an active run exists. The submit button becomes results-only when
the catalog's `submission_supported` field is false; each harness also exposes
sanitized `available` status and a bounded unavailability reason. Active runs
can be cancelled from their expanded detail.

All routes require a Nexus session. POST routes also use Nexus's normal write
session, CSRF, Origin, and audit controls:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/benchmarks/catalog` | reviewed suites, versions, combinations, and readiness |
| `GET` | `/api/benchmarks/runs?limit=100` | sanitized newest-first summaries |
| `GET` | `/api/benchmarks/runs/{run_id}` | verified case evidence for one run |
| `POST` | `/api/benchmarks/runs` | idempotently submit an exact combination |
| `POST` | `/api/benchmarks/runs/{run_id}/cancel` | request cancellation with `{"confirmed":true}` |

A submission body is shaped as follows; IDs must come from the catalog:

```json
{
  "request_id": "br-00000000000000000000000000000000",
  "suite_id": "smoke",
  "harness_id": "opencode",
  "model_id": "local/qwen",
  "tool_profile_id": "fleet-local",
  "repetitions": 1
}
```

`request_id` is the idempotency key for one logical submission. Reusing it with
the exact same normalized request returns the existing run instead of starting
duplicate work; reusing it for a different request is rejected fail-closed.
Repetitions are bounded to 1 through 20. Browser clients receive no raw prompt,
credential, endpoint, command, model path, Fleet ticket, claim, UUID, device,
PID, or tool output.

## CLI and durable state

The CLI uses `~/.local/share/aeon/benchmarks` by default, or an absolute
owner-private root supplied by `AEON_BENCHMARK_HOME` or `--root`:

```bash
cd /home/aday/NexusAgentDashboard/bc_aeon
python3 -m aeon.benchmarks catalog
python3 -m aeon.benchmarks list --limit 20
python3 -m aeon.benchmarks show run-0123456789abcdef0123456789abcdef
python3 -m aeon.benchmarks cancel run-0123456789abcdef0123456789abcdef
python3 -m aeon.benchmarks submit \
  --suite smoke \
  --harness opencode \
  --model local/qwen \
  --tool-profile fleet-local \
  --repetitions 1
```

Nexus uses its own owner-private `state_dir/benchmarks` root. SQLite state and
evidence are deliberately separate from source. Per-run harness workspaces are
also owner-private and retained for narrow manual cleanup. Each harness receives
a separate owner-private state sibling, so OpenCode configuration, session, and
supervisor data cannot overlap the model-facing case workspace. The runner does
not recursively delete canonical-host state. Directories must be mode `0700`;
the database and evidence files must be singly linked owner files with mode
`0600`. Each terminal run publishes bounded JSON evidence containing only
sanitized case fields, then binds it to an `evidence_sha256`. Readback verifies
ownership, size, shape, run identity, and digest before returning cases. Failed
verification yields no untrusted cases.

The worker registers its exact owner PID and kernel start time only in the
private queue database. Read paths reconcile an active run whose bound process
has disappeared to `failed/worker_lost`; an unregistered queue receives a
bounded startup grace before the same truthful transition. PIDs and process
identity never enter the Nexus response or evidence.

Provenance records the catalog, suite, request, runner protocol, exact
combination, evidence, and content-bound executor, runner, harness, and tool
source SHA-256 values, plus harness version, model revision, and tool-profile
version. These digests bind the trusted prompt/scoring implementation, durable
queue and timing logic, both harness launch paths, the complete reviewed tool
profile, Fleet-wait bridge, and immutable browser fixtures without exposing
their contents. A one-byte change in any bound source scope changes its digest,
and a queued run refuses execution after drift. Compare runs only when the
relevant identities match.
The current model revision names a logical service and deliberately does not
attribute a concrete runtime. After a catalog, prompt, fixture, scoring,
result-field, harness, model, or tool
change, the corresponding version/hash must change too.

## Deadlines, cancellation, and cleanup

The outer worker and every harness subprocess are launched shell-free with fixed
argv. Captured output is capped at 4 MiB. Each harness owns exactly one normal
`BrokerServiceSession`; no benchmark probe, availability scanner, second ticket,
or warm reservation is created. Sanitized `waiting_for_compute` and `allocated`
transitions travel back through an inherited anonymous pipe. Only a broker-proven
wait pauses the case-execution clock and changes the durable run state to
`waiting_for_compute`; startup, model work, tools, and verification still consume
the deadline. Total wall time continues to include the wait, while active time
excludes only measured broker-proven intervals so queue pressure does not skew
harness comparisons.

The runner polls the durable cancel bit even while compute is unavailable.
Deadline, cancellation, or overflow terminates the exact task-owned process tree
and revalidates each PID against its kernel start time before signaling; this
includes OpenCode's separate session and MCP child. It waits for proven exit,
escalates only within that exact tree, and records `stuck` if termination cannot
be proved. Normal signal unwinding releases the exact Fleet ticket; a failed
release is never reported as successful cleanup.

Cancelling a queued run makes it terminal immediately. Cancelling during a run
stops the current exact harness, prevents later cases, and preserves already
completed sanitized evidence. Harness cleanup remains responsible for releasing
its exact Fleet service ticket. Never kill a benchmark by process name or infer
GPU availability from processes.

## Adding a combination or case

1. Add a stable `HarnessDefinition`, `ModelSpec`, or `ToolProfileSpec` in the
   owning server catalog. For a harness or model, first satisfy the runtime and
   Fleet contracts in [OPENCODE_HARNESS.md](OPENCODE_HARNESS.md).
   A concrete model choice additionally requires a broker-supported per-demand
   routing constraint and execution identity evidence; changing the shared global
   service preference for a benchmark is not an acceptable substitute.
2. Add only explicitly compatible `CombinationSpec` entries with immutable
   harness, model-revision, and tool-profile versions. Unknown combinations must
   remain rejected.
3. Define cases as semantic `ScenarioSpec` IDs. Keep actual prompts, fixture
   paths, credentials, and execution details inside the trusted executor; they
   must never appear in the public catalog or submitted request.
4. Add a bounded executor branch that invokes the real reviewed harness and
   independently verifies the intended effect. A model saying "done" is not
   sufficient for a file, browser, vision, or tool success.
5. Give every new external dependency a fail-closed readiness check. Browser
   cases must use an immutable authenticated internal fixture rather than an
   arbitrary third-party site or caller-supplied content.
6. Bump suite/catalog versions for case or fixture changes. Bump the runner
   protocol when result fields or scoring semantics change. Update the expected
   hashes and add hermetic refusal, timeout, cancellation, cleanup, sanitization,
   tamper, and idempotency tests.

## Verification and deployment

The benchmark regressions use fake process/model/Fleet boundaries and do not
acquire live GPU compute:

```bash
cd /home/aday/NexusAgentDashboard/bc_aeon
python3 -m pytest -q aeon/tests/test_benchmarks.py
```

Run the Nexus API and browser/UI benchmark tests from `dashboard` as part of a
cross-project change:

```bash
cd /home/aday/NexusAgentDashboard/dashboard
python3 -m pytest -q tests/test_app.py -k benchmark
python3 -m pytest -q tests/test_frontend_readability.py -k benchmark
python3 -m pytest -q tests/test_browser_e2e.py -k benchmark
```

Follow
`/home/aday/NexusAgentDashboard/fleet_compute/docs/INTEGRATED_OPERATIONS.md` for
deployment. Benchmark Python is loaded by newly spawned workers, while the
catalog/API and static page are held by the long-running Nexus process. Restart
only `nexus-backend.service` after tests, refresh the browser, and verify the
catalog plus a bounded run. Do not restart Fleet Compute unless its own enabled
profile, adapter, or entry-point contract changed.
