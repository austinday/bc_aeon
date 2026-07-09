# Aeon

Aeon is an autonomous, self-modifying agent harness. A single LLM drives a
plan→act→observe loop with collapsible tools, reusable skill protocols,
parallel sub-agents, persistent memory, and live context-pressure management.
The same repo is portable across machines: local models are auto-deployed to fit
the detected GPUs, and cloud models (Gemini/Vertex/Grok) work out of the box.

## Quick start

```bash
pip install .                      # installs the `aeon` console script
python3 -m aeon.main               # interactive start: model picker (Enter = Qwen3.6-27B-FP8, solo GPU0)
python3 -m aeon.main --model Qwen3.5-397B-A17B-Q3K   # skip the picker, name a model directly
python3 -m aeon.main -n --start "Do X"               # headless: no picker, boots the default model
```

Common flags:

```bash
python3 -m aeon.main --start "Summarize this repo"
python3 -m aeon.main --start "Build X and test it" --max-iterations 40
python3 -m aeon.main --help        # full flag list
```

| Flag | Purpose |
|------|---------|
| `--model NAME` | Skip the menu and use a specific model |
| `--menu` / `-i` | Force the interactive model picker (choose solo vs dual-GPU, etc.) |
| `--dual` | Deploy the main model across BOTH GPUs (a copy per GPU + routing) instead of the default single-GPU/solo placement |
| `--start "..."` | Run an objective immediately, then drop into the REPL |
| `--max-iterations N` | Cap iterations per objective (forces a final report at the limit) |
| `--no-warmup` | Skip model warmup (faster startup) |
| `--debug` / `--debug-log PATH` | Verbose LLM/reasoning logging |

## How the loop works

Each iteration the harness assembles one prompt containing: core directives,
the visible tools, active skill protocol, persistent memories, a compressed
attempt log, runtime/GPU stats, the project tree, open files, the last result,
and a live sub-agent digest. The model replies with a JSON action plan
(`thought`, `intent`, `updated_plan`, `actions`, plus mandatory `skill_check` /
`memory_check` / `parallel_check` reflections). Multi-line content is passed via
`--- BEGIN BLOCK_N ---` blocks appended after the JSON, so code never has to be
escaped into JSON strings.

Built-in robustness:

- **Context pressure** is estimated every turn; the attempt log and memories are
  auto-compressed, and at >95% of the limit the largest/oldest open files are
  shed automatically instead of crashing.
- **Cross-run persistence**: memories, the plan, and the attempt log are written
  to `aeon_output/session_state.json` after every iteration. A fresh process
  restores durable memories (and, when the objective matches, the plan and log)
  instead of starting from amnesia — surviving a crash or clean exit, not just an
  in-process `restart_aeon`.
- **Verified edits**: `write_file` / `str_replace` return a compact unified diff
  of what actually changed (so a fuzzy/whitespace match landing in the wrong
  region is visible), plus a non-blocking syntax warning when a written
  `.py`/`.json`/`.yaml` file no longer parses.
- **Loop & stall detection**: identical command+output repeats, and the same
  intent repeated across turns, both trigger corrective nudges.
- **JSON recovery**: malformed output is repaired locally (trailing commas,
  Python literals) before falling back to a model-based repair.
- **Tool calls** auto-correct trivial name typos, suggest close matches on a
  miss, and report the expected signature on a bad-parameter call.
- **`run_command`** streams output, enforces its timeout even on silent hangs,
  and kills the whole process group (no orphaned GPU/CPU jobs).

## Tools and skills

Tools are grouped into collapsible categories to save context; top-level tools
(`run_command`, `open_file`, `write_file`, `str_replace`, `think`, `memorize`,
`spawn_sub_agent`, …) are always visible, and the agent calls
`expand_tool_category` / `collapse_tool_category` to reveal the rest (image,
video, browser, sub-agent coordination, …). Skills are pinned step-by-step
protocols the agent activates for matching objectives.

## Self-modification & self-improvement

Aeon can edit its own source, restart onto the new code, and measure whether the
change actually made it better — a guarded, reversible loop rather than a blind
overwrite.

- **Author skills by asking (full CRUD)**: `create_skill` (add), `read_skill`
  (inspect without activating), `create_skill(..., overwrite=true)` (modify), and
  `delete_skill` (remove). Every change is live the SAME session (skills are read
  from disk each turn — no restart for a skill that needs no new tool) and persists
  across restarts.
- **Add tools by asking**: drop a `BaseTool` subclass under `aeon/tools/` and call
  `restart_aeon`; the dynamic loader picks it up.
- **Zero-path self-modification**: `restart_aeon` and `verify_self_modification`
  auto-derive the agent's own source root from the installed package, so a
  self-modification "just works" from any workspace — the model never has to know
  or hand-supply the path to its own code, and verification always installs/tests
  the aeon source (not whatever project directory it happens to be running in).

- **Durable git checkpoints.** Every `restart_aeon` first tags the working tree as
  a recoverable checkpoint (`aeon/core/checkpoint.py`), capturing tracked *and*
  untracked files. Restores reconcile modifications, deletions, and additions so
  the `aeon/` package matches the checkpoint exactly, and never touch files
  outside it. Checkpoints persist as a diffable lineage (not a one-shot tarball).
- **Boot handshake (no bricking).** The pip/smoke/unit gates run the new code as a
  *subprocess*, but `execv` relaunches through the untested `--resume` path. A
  pending-boot marker (`aeon/core/bootguard.py`) names the checkpoint to roll back
  to; the relaunched process clears it only once it boots healthy, and any fresh
  start that still sees it auto-reverts. A resume-path crash can no longer leave
  broken code installed with no way back.
- **`revert_aeon`.** An agent-callable rollback to any checkpoint, for the case the
  gates can't catch: a change that passes tests but, once live, is a behavioural
  regression. Roll back, restart, continue.
- **Protected core.** A small constitution (`aeon/core/protected.py`) — the
  benchmark/scorer, the checkpoint & rollback engine, the boot handshake, the test
  gates — is refused by the edit tools unless a human sets
  `AEON_ALLOW_PROTECTED_EDIT=1`, so a self-modification can't quietly weaken the
  machinery that measures and reverts it.
- **A fitness signal.** `aeon/selfimprove/` turns one-off editing into a measurable
  loop: a scored **benchmark** of capability tasks run against a candidate in an
  isolated sandbox copy (`evaluate.py`), a **scorer** that accepts a change only if
  it holds or improves the score with no regressions (the ratchet), and a durable
  experiment **ledger**. `run_self_benchmark` exposes the score to the agent, so it
  can baseline before a change and confirm improvement (or trigger `revert_aeon`)
  after. Deterministic tasks need no model; model-driven behavioural tasks plug
  into the same registry.

## Web browsing (human-grade)

The `web_browser` tools drive **real Google Chrome** (not Chromium) via
**Patchright** (a patched Playwright that removes the CDP automation tells most
bot-detectors probe for), running **headed** under Xvfb (at a real-world
1920×1080) in a container (`aeon/services/browser/`) with a **persistent profile**
so logins/cookies survive. It uses no spoofed user-agent/viewport and no
detectable evasion shims; **every** action is physically human — the cursor
follows curved, time-sampled Bézier paths with an accelerate-then-settle velocity
and slight overshoot, clicks land on a randomized point within the element (not
dead center), keystrokes type at a real ~110 WPM cadence with word-boundary pauses
and occasional hesitations, scrolling rolls in wheel notches, drags
press-pause-move-release, checkboxes/switches are *clicked* (not toggled
programmatically), dropdowns are clicked open before selecting, fields are cleared
with select-all+delete, and even the pointer drifts a little while "reading."
Combined with the host's residential IP, the goal is to be indistinguishable from
a person.

Perception is **first-person**: the agent *is* the multimodal model (Gemma-4), so
each browser action attaches the **actual rendered screenshot** to its next turn —
it looks at the page exactly as a human would and decides from the pixels, not
from a secondhand text caption. That view is paired with a **stable, indexed
element list** built from the accessibility tree — each interactable/meaningful
node stamped with a `data-aeon-id` and described by its role, accessible name,
value, and state (`expanded`/`collapsed`, `selected`, `checked`, `disabled`,
off-screen, scroll-group). So the model sees the page *and* has exact `[id]`s to
act on (no selector guessing). By default the screenshot carries small numbered
`[id]` marks (Set-of-Mark, for precise grounding); `visual="clean"` gives the pure
render, `visual="both"` gives both, and `include_vision=false` skips the image for
a faster element-list-only turn.

Actions (`browser_interact`) cover the full human repertoire by id: click /
double / right-click, hover, type (real keystrokes, optional submit), press_key,
**scroll** (page or within a specific scroll container — e.g. an inbox list),
drag, press-and-hold, select_option, check/uncheck, upload_file, back/forward/
reload, wait_for, and `read_text` (clean readability extraction). Clicks can pass
`expected_text`, which is verified against the target before clicking to prevent
wrong-element clicks. `browser_read` re-observes without acting.

It also handles the things real sites throw at you: the element index descends
into **iframes and shadow DOM** (marked «in iframe»); native JS **dialogs** are
handled — but a `confirm`/`prompt` that looks **destructive** (delete/discard/
overwrite…) is *dismissed*, not auto-confirmed, and reported, so the agent never
silently deletes something; **downloads** are captured and copied into the
workspace `./downloads` so the agent can use them; **PDFs** (whose text isn't in
the DOM) are fetched and saved for parsing on `read_text`; button-triggered **file
pickers** work for uploads; **popups/OAuth windows** are captured as switchable
tabs; and a crashed browser is transparently relaunched.

More that keeps it human and capable:
- **Identity matches the network.** On launch the browser's timezone, locale, and
  `navigator.languages` are set from the **egress IP's geolocation** (with matching
  geo-coordinates), so an IP-vs-clock/language mismatch — a top bot signal — never
  shows. Override with `AEON_BROWSER_TZ` / `AEON_BROWSER_LOCALE`.
- **Proxy + leak protection.** Set `AEON_BROWSER_PROXY`
  (`http://user:pass@host:port` or `socks5://…`) to appear from anywhere; WebRTC
  is prevented from leaking the real IP around the proxy.
- **Idle motion + adaptive timing.** The cursor makes small "reading" drifts
  between actions (behavioral realism), and each action waits only until the DOM
  actually settles (fast on stable pages, patient on churning ones) instead of a
  fixed sleep.
- **See what changed.** `browser_interact(..., compare=true)` attaches the
  before *and* after screenshots so the model can diff its own action.
- **Per-agent isolation.** Each agent gets its OWN browser context (own cookie
  jar, storage, history, fingerprint): the principal uses the persistent `default`
  profile so logins survive across runs, while each sub-agent browses in an
  isolated context (`agent-<id>`) — so parallel agents are independent identities
  that never collide, and a sub-agent's context is torn down when it finishes.
- **Full navigation.** Multiple named tabs with switch/close; site-opened
  tabs/popups (target=_blank, window.open, OAuth) are captured, announced, and
  usable/closable; self-closing popups (OAuth) are detected and pruned; scrolling
  works both vertically AND horizontally, page-level or within a specific pane.
- **Challenges.** A `web/solving_challenges` skill guides the vision model through
  CAPTCHAs, image grids, sliders, and verification walls the human way.

## Self-modification

Aeon can edit its own source. The workflow is: change the code →
`verify_self_modification` (sandboxed sub-agent test) → `restart_aeon`. A restart
backs up the code, reinstalls, and **only relaunches if both the smoke test and
the unit tests pass**; otherwise it restores the backup and keeps the old code
running. All additions must be portable — Aeon runs from any project directory
and must not depend on its own source being in the workspace.

## Testing

```bash
./run_tests.sh                     # smoke test + unit tests (no GPU/model needed)
python3 -m aeon.smoke_test         # imports, tool discovery, syntax
python3 -m aeon.tests.test_core    # pure-logic unit tests
```

These are the same gates the restart path runs, so they are the fast way to
confirm a change is safe before going live.
