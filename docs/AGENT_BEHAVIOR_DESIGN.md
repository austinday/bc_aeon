# Aeon behavior design

## Objective

Aeon should take the shortest evidence-producing route on an ordinary task, keep
working through recoverable failures, change abstraction level when a narrow route
stalls, and claim completion only when owner-authored outcomes are verified. No
agent can be guaranteed to succeed on every task; the harness instead guarantees
bounded execution, explicit recovery, evidence-backed completion, and a truthful
typed terminal state.

This design deliberately avoids adding unconditional “reflection” turns. More
reasoning is useful only when the current trajectory supplies a reason to spend it.

## Findings from the prior loop

The previous behavior coupled several distinct questions:

- Did a tool call fail?
- Is this exact call worth retrying?
- Has the current method failed?
- Is the parent goal impossible?
- Did activity satisfy the user's request?

That coupling produced both premature surrender and loops. A failure could become a
terminal blocker immediately, while a superficially different command could reset
the loop detector. Model-authored plans acted like completion state, successful but
irrelevant reads looked like progress, generic tests could validate unrelated edits,
and context projection discarded the strategic history needed to avoid old routes.
Polite actionable requests were sometimes treated as hypothetical plans, whereas a
model could ask “Should I proceed?” after already receiving exact authority.

## Research synthesis

The implementation follows these recurring results from agent research:

- Tool interface and observation design materially affect coding-agent performance,
  not just the base model ([SWE-agent](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html)).
- Interleaving grounded observations with action selection is more reliable than a
  long detached plan ([ReAct](https://arxiv.org/abs/2210.03629)), while planner/worker
  separation can reduce redundant observation cost ([ReWOO](https://arxiv.org/abs/2305.18323)).
- Adaptive decomposition and complexity routing are preferable to paying the same
  reasoning cost for every task ([ADaPT](https://aclanthology.org/2024.findings-naacl.264/),
  [E3](https://www.microsoft.com/en-us/research/publication/do-ai-agents-know-when-a-task-is-simple-toward-complexity-aware-reasoning-and-execution/),
  [Agentless](https://arxiv.org/abs/2407.01489)).
- Ungrounded self-critique is unreliable on its own; correction improves when it is
  anchored in tool/environment feedback
  ([self-correction limits](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html),
  [CRITIC](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html),
  [Reflexion](https://papers.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)).
- Long contexts need deliberate projection and durable state because relevant facts
  are not used uniformly across their position
  ([Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/),
  [Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).
- Final success scores hide lucky passes and failure mechanics. Evaluation should
  inspect trajectories and intermediate progress
  ([AgentBoard](https://proceedings.neurips.cc/paper_files/paper/2024/hash/877b40688e330a0e2a3fc24084208dfa-Abstract-Datasets_and_Benchmarks_Track.html),
  [AgentRx](https://www.microsoft.com/en-us/research/publication/agentrx-diagnosing-ai-agent-failures-from-execution-trajectories/),
  [AgentLens](https://www.microsoft.com/en-us/research/publication/agentlens-revealing-the-lucky-pass-problem-in-swe-agent-evaluation/)).
- Stateful tool use must preserve authority and exact targets across multi-turn
  interactions ([ToolSandbox](https://arxiv.org/abs/2408.04682),
  [Berkeley Function-Calling Leaderboard](https://proceedings.mlr.press/v267/patil25a.html),
  [SABER](https://www.amazon.science/publications/saber-small-actions-big-errors-safe-guarding-mutating-steps-in-llm-agents)).

## Implemented control architecture

### Complexity-routed fast path

Simple inspection/extraction turns use low effort and one proposal. Established
execution uses medium effort. A complex first decomposition uses high effort.
Ordinary recovery stays at medium effort on one coherent trajectory; only a
level-three parent-route reframe activates high effort. Multi-candidate sampling is
an explicit operator override rather than an automatic latency multiplier. Every
decision retains a hard call/token/time budget.

### Evidence-triggered recovery ladder

The deterministic progress controller distinguishes exact actions, structural
method families, and outcomes.

1. First failure: check a missing precondition or choose a different action.
2. Repeated/equivalent failure: bar the exact call and require a different method
   family.
3. A/B oscillation or broad plateau: choose a different route to an unmet owner
   outcome.

Recovery text is a compact action constraint. It explicitly forbids narrating the
recovery, restating the objective, or rewriting an unchanged plan; the harness owns
the diagnosis and the model owns only the next evidence-producing decision.

Parameter churn and rewording do not count as strategy diversity. The harness stops
only after two ignored exact-call bars or a bounded multi-method plateau. Only typed
acceptance/evidence progress resets the epoch.

### Owner-goal evidence graph

Conservative clause compilation separates aggregate, change, inspection,
validation, and invariant goals. The harness infers omitted action bindings from
typed targets and receipts; optional explicit leaf IDs disambiguate difficult cases.
Change goals need a relevant mutation and validation after the final mutation;
inspection goals need relevant information; validation goals need a relevant
outcome check; invariants observe all mutations. A model-authored checklist is
coordination state, never proof.

### Strategic continuity without memory coupling

A bounded session ledger records factual method families, goal IDs, and typed
outcomes. It survives history projection and process restore, but contains no hidden
chain-of-thought and does not read or write Aeon's memory subsystem. Replacement or
revoked authority clears the request-local ledger; additive work preserves it.

### Narrow questions and exact authority

The harness rejects generic permission-seeking once a request is actionable. It
allows a question only for a concrete unresolved value that cannot safely be
discovered. Continuations preserve confirmation, addition, replacement, revocation,
and exact target correction. External actions bind operation and recipient,
platform, site, or account; recovery cannot substitute a different destination.

### Bounded delegation and reads

Read-only transient errors receive one cancellable exact retry. Sub-agents enter
their isolated workspace before worker/tool construction. Report previews do not
launder collection; offset pagination reaches EOF before a report is complete.
Prose reports cannot validate principal mutations. Mutable children produce an
immutable, receipt-bound patch; only the principal can integrate it, after exact
process absence, base/conflict, workspace, protected-path, invariant, and final-hash
checks. Parallel read batches stop when cancellation or new user input arrives.

## Behavioral evaluation

The regression suite uses scripted models and fake tools to score outcomes rather
than prompt wording. It includes:

- failure → diagnosis → different mechanism → targeted validation → completion;
- A/B oscillation → parent-goal reframe → independent route;
- two ignored exact-call bars → typed bounded stop;
- false blocker and false completion rejection;
- generic permission-question rejection and concrete missing-input continuation;
- multi-goal evidence isolation, validation invalidation, and invariant enforcement;
- authority confirmation/revocation/replacement and restore;
- candidate/verifier budget exhaustion with a valid-proposal fallback;
- strategy-ledger and recovery-state restoration;
- sub-agent workspace isolation, report pagination, and conflict-safe changeset
  integration that still requires principal-state validation.

Operational metrics worth tracking in live traces are turns to first useful evidence,
same-method retry rate, unsupported ask-user rate, recovery success by level,
completion-claim rejection rate, validator relevance, median tool/model calls per
completed task, and terminal-state precision. These expose roundabout or lucky-pass
behavior much earlier than a single pass/fail score.
