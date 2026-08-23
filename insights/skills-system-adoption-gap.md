# Skills system: built out, but no evidence it is ever used

## Summary

The skills subsystem is now substantial — 29 protocols across 13 categories
(~21.8k tokens of curated procedure text), a full CRUD tool suite
(activate/deactivate/create/read/delete), a turn-1 LLM router, and a per-turn
`skill_check` reflection field. Yet across **every log and recorded output in
this repo** (aeon.log May 27 → Jul 14, 86 sub-agent output/telemetry/agent.log
files) there is **zero evidence of a single skill activation or routing hit**.
The last three commits all grow the skill library; nothing measures whether it
gets consumed.

Caveat: activation prints to console only (see Observability below), so absence
from logs is weak-but-only-available evidence. It is consistent with the code
reading: each of the three delivery mechanisms has a specific defect that biases
the agent toward never activating.

## Mechanism-by-mechanism analysis

### 1. Per-turn `skill_check` cannot see the skills

`_get_skills_description()` (worker.py:346) renders every category **collapsed
by default** (`expanded_categories` starts empty, worker.py:118):

```
[+] coding: (3 skills)
[+] compbio: (3 skills)
...
```

The instruction (primary_agent_instructions.txt:49) asks the model each turn to
"state whether the current objective matches a protocol in the SKILLS section"
— but the section shows only 13 category names and counts. The model literally
cannot match against skill names/descriptions it was never shown, unless it
first spends a turn on `expand_tool_category`, which nothing prompts it to do.

### 2. Both JSON examples anchor `skill_check` to "No matching skill."

The two full example responses in primary_agent_instructions.txt (lines 67 and
81) both contain `"skill_check": "No matching skill."`. For a local model
emitting rigid JSON every turn, few-shot anchoring dominates: copying the
example string verbatim is the path of least resistance, turning the reflection
field into boilerplate. (This is the same class of problem as the earlier
`__BLOCK_N__` artifacts — the model reproduces scaffolding text literally.)

### 3. The router's directive is ephemeral

`route_skills()` (llm.py:159) runs once per objective and its `[SKILL ROUTING]`
directive is appended to `self.last_observation` (worker.py:1789).
`last_observation` is **overwritten after the first action executes**. If turn
1's response is anything other than an immediate `activate_skill` (typical
turn-1 behavior is plan-writing plus a first exploratory command), the routing
hint vanishes from all subsequent prompts and never reappears.

### 4. Zero observability

- `activate_skill` / `deactivate_skill` / router hits go to `print()` only —
  nothing reaches the `aeon` logger or any persisted artifact.
- `route_skills` logs only failures (`logger.warning`), never successes.
- Consequence: the project cannot answer "are skills used?" from its own data.
  This analysis had to infer it negatively.

## Cost analysis (measured)

| Item | Tokens (chars/4 estimate) |
|---|---|
| Collapsed SKILLS section, paid every turn | ~105 |
| Router catalog prompt, once per objective | ~2,000 |
| Pinned protocol while active (mean / max) | ~750 / ~1,520 per turn |
| Full flat catalog with one-line descriptions | ~1,900 |

The striking number: showing the **entire flat catalog** (29 lines, router-style
descriptions) costs ~1.9k tokens — static text, sits in the cacheable prefix,
prefilled once per session under vLLM prefix caching. The current design hides
skills to save ~1.8k effectively-free tokens, then spends a full big-model call
(the "utility" client aliases the primary model, llm.py:59) per objective to
compensate for the hiding.

## Recommendations, in order of leverage

1. **Un-collapse the catalog.** Render all 29 skills flat with their router
   descriptions in the cacheable prefix. This gives the per-turn `skill_check`
   something to actually match against and makes the router mostly redundant.
2. **Make the routing directive sticky.** Store it like `active_skill`
   (e.g. `self.pending_skill_suggestion`) and re-inject each turn until the
   agent either activates the skill or explicitly rejects it in `skill_check`,
   then clear it.
3. **De-anchor the examples.** Make one of the two instruction examples show a
   positive check (`"skill_check": "Matches coding/debugging_root_cause —
   activating"` with `activate_skill` as the first action).
4. **Log skill lifecycle events.** `logger.info` on route hit, activation,
   deactivation, creation, deletion. One line each; enables the adoption metric
   this report could not compute.
5. **Then measure.** After 1–4, count activations per objective across a week of
   real sessions. If protocols still go unused, the library is dead weight and
   the honest move is to fold the best content into core directives.

## Attempted live router benchmark (inconclusive)

A 31-case accuracy benchmark of the exact `route_skills` prompt (17 clear
matches, 5 should-be-NONE, 9 ambiguity probes) was run against ollama
llama3.1:8b as a stand-in utility model, but was abandoned after 68 minutes:
with the GPUs saturated by training jobs, each ~2k-token routing call took
minutes. Rerun it when the aeon vLLM server is up by pointing the same catalog
+ prompt construction (llm.py:180-204) at the live endpoint — routing accuracy
on the ambiguous pairs (deep_analysis vs codebase_analysis, NaN-loss →
troubleshooting vs machine_learning) is the number that decides whether the
router adds value over just showing the flat catalog.

## Minor related findings

- `coding/verifying_change_propagation` is the only skill whose description line
  gives no "use when" trigger (router sees just a title) — weakest routing odds.
- `SkillsManager()` is re-instantiated and the full skills tree re-read from
  disk in at least three places per turn (`_get_skills_description`, tools,
  router). Harmless at this scale (29 small files) but it means the "cacheable
  prefix" claim depends on files not changing mid-session — which is exactly
  what `create_skill` does. Acceptable; just worth knowing the cache busts once
  per skill mutation.
