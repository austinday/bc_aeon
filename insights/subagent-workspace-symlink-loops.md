# Sub-agent workspace symlinks create infinite traversal loops

## Finding

Every sub-agent gets a `workspace` entry that is a **symlink back to the
project root**:

```
aeon_output/<session>/sub_agents/<agent-id>/workspace -> /home/aday/bc_aeon
```

There are ~30+ of these accumulated in `aeon_output/`. Because the project root
*contains* `aeon_output/`, any recursive traversal that follows symlinks enters
a cycle: `bc_aeon → aeon_output → .../workspace → bc_aeon → ...`

This was discovered empirically during this analysis: a `glob.glob('aeon_output/**/*.json',
recursive=True)` ran until killed at the 2-minute timeout (Python's recursive
glob follows directory symlinks before 3.13). The directory "looks" like 63 MB /
187 files but is an unbounded tree to any follower of symlinks.

## What is and isn't affected

Aeon's own code is safe today — verified:

- Project tree builder (`system_info.py:165-167`) uses `is_dir(follow_symlinks=False)`.
- `_suggest_paths` (file_io.py:67) uses `os.walk` default `followlinks=False`.
- GNU `grep -r`, `find` default, `du` default do not follow symlinks.

But the hazard is one flag away, and the *agent itself* is the likely victim:
the primary agent and sub-agents routinely run ad-hoc shell commands and Python
snippets over the workspace (`grep -R`, `du -L`, `rsync -L`, `tar -h`, Python
`glob('**')`, `shutil.copytree`, `pathlib.rglob` on py<3.13 semantics, any
Node/JS tool with `followSymlinks: true`). Each accumulated workspace symlink
is a landmine for those. An agent that hangs on a traversal loop then trips the
stuck-detector, wastes iterations, or gets its command killed with no
explanation it can act on — this manifests as "mysterious slow/hung command,"
one of the loop-guard failure classes already seen in this project.

Secondary effect: sub-agents nominally get "Workspace set to read-only"
(agent.log), but the symlink points at the live read-write project root — the
isolation is nominal, and any recursive delete/copy a sub-agent runs in "its"
workspace operates on the real repo.

## Recommendations

1. **Stop symlinking the whole project root.** If sub-agents need repo access,
   pass the real path as cwd; a symlink inside their output directory adds
   nothing except the cycle.
2. If the symlink must stay, **name it outside the traversal namespace**
   (e.g. dot-prefixed `.workspace`, which most globs skip) or replace with a
   plain text file `workspace_path.txt` containing the path.
3. **Garbage-collect old sub-agent directories.** ~30 stale sessions of
   symlinks and logs accumulate risk and clutter; a startup sweep deleting
   `aeon_output/*/sub_agents/*` older than N days would cap it.
4. **Add a directive/skill note** warning the agent to avoid symlink-following
   flags when traversing the workspace (`grep -r` not `-R`, `find -P`, no
   `du -L`), since the loop exists in every historical session dir.

## Confirmed: read-only is not enforced at all

`sub_agent_wrapper.py` declares `--read_only` (line 65) and **never reads
`args.read_only` anywhere** — the flag is parsed and dropped. No permission
change, no sandbox, nothing instructional either in the current wrapper. The
"Workspace set to read-only." line in old agent.logs comes from a prior
version; today a "read-only" sub-agent has full write access to the live repo
through its workspace symlink. This mirrors the confabulation-class problem:
a claim in the system's own vocabulary that ground truth does not back.

Fix options: (a) drop the flag and the pretense; (b) enforce it for real —
run the sub-agent as cwd = a copy-on-write checkout (git worktree is cheap), or
inject a write-blocking directive plus wrap file-writing tools when the flag is
set. (a) is honest and one line; (b) is what the flag promises.
