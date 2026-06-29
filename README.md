# Aeon

Aeon is an autonomous, self-modifying agent harness. A single LLM drives a
plan→act→observe loop with collapsible tools, reusable skill protocols,
parallel sub-agents, persistent memory, and live context-pressure management.
The same repo is portable across machines: local models are auto-deployed to fit
the detected GPUs, and cloud models (Gemini/Vertex/Grok) work out of the box.

## Quick start

```bash
pip install .                      # installs the `aeon` console script
python3 -m aeon.main               # interactive: pick a model, then type objectives
```

Common flags:

```bash
python3 -m aeon.main --model gemini-flash-latest --start "Summarize this repo"
python3 -m aeon.main --start "Build X and test it" --max-iterations 40
python3 -m aeon.main --help        # full flag list
```

| Flag | Purpose |
|------|---------|
| `--model NAME` | Skip the menu and use a specific model |
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

## Web browsing (human-grade)

The `web_browser` tools drive **real Google Chrome** (not Chromium) via
**Patchright** (a patched Playwright that removes the CDP automation tells most
bot-detectors probe for), running **headed** under Xvfb in a container
(`aeon/services/browser/`) with a **persistent profile** so logins/cookies
survive. It uses no spoofed user-agent/viewport and no detectable evasion shims —
combined with human-like mouse/keyboard input and the host's residential IP, the
goal is to be indistinguishable from a person at a normal browser.

Every observation gives the agent two aligned channels:
- a **stable, indexed element list** built from the accessibility tree — each
  interactable/meaningful node is stamped with a `data-aeon-id` and described by
  its role, accessible name, value, and state (`expanded`/`collapsed`, `selected`,
  `checked`, `disabled`, off-screen, scroll-group). The agent acts on elements by
  `[id]`, which resolves to the exact node (no selector guessing).
- a **Set-of-Mark screenshot** with numbered boxes that match those same ids,
  analyzed by the vision model.

Actions (`browser_interact`) cover the full human repertoire by id: click /
double / right-click, hover, type (real keystrokes, optional submit), press_key,
**scroll** (page or within a specific scroll container — e.g. an inbox list),
drag, press-and-hold, select_option, check/uncheck, upload_file, back/forward/
reload, wait_for, and `read_text` (clean readability extraction). Clicks can pass
`expected_text`, which is verified against the target before clicking to prevent
wrong-element clicks. `browser_read` re-observes without acting.

It also handles the things real sites throw at you: the element index descends
into **iframes and shadow DOM** (marked «in iframe»); native JS **dialogs**
(alert/confirm/prompt/beforeunload) are auto-handled and reported; **downloads**
are captured to `~/.aeon/browser_profiles/downloads` and their path reported;
button-triggered **file pickers** work for uploads; **popups/OAuth windows** are
captured as switchable tabs; and a crashed browser is transparently relaunched.

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
