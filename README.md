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
