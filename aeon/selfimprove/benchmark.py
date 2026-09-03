"""Scored capability tasks — the fitness signal for self-improvement.

Each task runs INSIDE a candidate's code (via ``runtask`` with the candidate on
PYTHONPATH) and returns pass/fail plus an optional metric. The point is to give
self-modification a number it can move: a change that breaks tool loading, the
edit tools, the protected-core guard, or the rollback engine makes the score
drop and is rejected by the ratchet, while a genuine improvement holds or raises
it.

These deterministic tasks need no model/GPU, so the harness has a real fitness
signal today. Model-driven behavioural tasks (run an objective end-to-end, check
the deliverable) plug in through the same registry under ``kind="agent"`` and are
skipped when no model is configured.

A task function returns: (passed: bool, detail: str, metric: float | None).
"""


def _t_tools_import_clean():
    """Every tool module imports and every dependency-satisfied tool instantiates."""
    from aeon.tools.loader import load_tools_from_directory

    class _W:  # minimal stand-in for the worker dependency
        def __init__(self):
            self.open_files = {}
            self.memories = {}
        def is_file_open(self, p): return False
        def close_file(self, p): return True
        def update_open_file(self, p, c): self.open_files[p] = c

    errors = []
    tools = load_tools_from_directory("aeon.tools", dependencies={"worker": _W()}, errors_out=errors)
    if errors:
        return False, f"{len(errors)} loader error(s): {errors[0]}", float(len(tools))
    if not tools:
        return False, "no tools loaded", 0.0
    return True, f"{len(tools)} tools loaded cleanly", float(len(tools))


def _t_edit_tools_roundtrip():
    """Writes are hash-bound and syntactically invalid content is rejected."""
    import hashlib
    import os
    import tempfile
    from aeon.tools.file_io import WriteFileTool, StrReplaceTool

    class _W:
        def __init__(self, root):
            self.open_files = {}
            self.workspace_root = root
            metadata = os.stat(root, follow_symlinks=False)
            self.workspace_root_identity = (metadata.st_dev, metadata.st_ino)
        def is_file_open(self, p): return os.path.abspath(p) in self.open_files
        def close_file(self, p): self.open_files.pop(os.path.abspath(p), None); return True
        def update_open_file(self, p, c): self.open_files[os.path.abspath(p)] = c

    d = tempfile.mkdtemp()
    w = _W(d)
    wf, sr = WriteFileTool(w), StrReplaceTool(w)
    path = os.path.join(d, "m.py")
    if "Created" not in wf.execute(path, "def f():\n    return 1\n"):
        return False, "write_file create failed", None
    with open(path, "rb") as source:
        receipt = hashlib.sha256(source.read()).hexdigest()
    r = sr.execute(
        path, old_str="return 1", new_str="return 2", expected_sha256=receipt
    )
    if "Successfully applied" not in r or "DIFF" not in r:
        return False, f"str_replace/diff failed: {r[:120]}", None
    with open(path) as f:
        if "return 2" not in f.read():
            return False, "edit not persisted", None
    broken = wf.execute(os.path.join(d, "b.py"), "def x(:\n    pass\n")
    if "Refusing" not in broken or os.path.exists(os.path.join(d, "b.py")):
        return False, "invalid Python was not rejected before write", None
    return True, "hash-bound edit + diff + pre-write syntax gate intact", None


def _t_protected_guard_active():
    """The protected-core guard blocks a guarded file and permits a normal one."""
    from aeon.core import protected
    blocked = protected.guard(protected.__file__)  # the guard protects itself
    if not blocked:
        return False, "guard did NOT block its own protected file", None
    allowed = protected.guard("/tmp/some_random_user_file.py")
    if allowed is not None:
        return False, "guard wrongly blocked a non-protected file", None
    return True, "protected-core guard active", None


def _t_checkpoint_engine_intact():
    """Checkpoint create+restore works in a throwaway git repo (rollback engine sane)."""
    import os
    import subprocess
    import tempfile
    from aeon.core import checkpoint

    d = tempfile.mkdtemp()
    def git(*a):
        return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)
    git("init", "-q")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    pkg = os.path.join(d, "aeon"); os.makedirs(pkg)
    fpath = os.path.join(pkg, "f.txt")
    with open(fpath, "w") as f:
        f.write("v1\n")
    git("add", "-A"); git("commit", "-qm", "init")
    if not checkpoint.is_git_repo(d):
        return False, "is_git_repo false in a real repo", None
    ck = checkpoint.create_checkpoint(d, "test")
    if not ck.get("ok"):
        return False, f"create_checkpoint failed: {ck.get('reason')}", None
    with open(fpath, "w") as f:
        f.write("v2-broken\n")
    res = checkpoint.restore_checkpoint(d, ck["tag"])
    if not res.get("ok"):
        return False, f"restore failed: {res.get('reason')}", None
    with open(fpath) as f:
        if f.read().strip() != "v1":
            return False, "restore did not bring back original content", None
    return True, "checkpoint create/restore intact", None


# id -> (kind, weight, fn, description)
TASKS = {
    "tools_import_clean":     ("deterministic", 1.0, _t_tools_import_clean,
                               "All tool modules import and instantiate cleanly."),
    "edit_tools_roundtrip":   ("deterministic", 1.0, _t_edit_tools_roundtrip,
                               "write_file/str_replace + diff + syntax check work end to end."),
    "protected_guard_active": ("deterministic", 1.0, _t_protected_guard_active,
                               "Protected-core guard blocks guardrail edits, allows normal edits."),
    "checkpoint_engine_intact": ("deterministic", 1.0, _t_checkpoint_engine_intact,
                               "Git checkpoint create/restore (the rollback engine) works."),
}


def deterministic_ids():
    return [tid for tid, (kind, *_rest) in TASKS.items() if kind == "deterministic"]


def run_task(task_id: str):
    """Execute one task by id; return (passed, detail, metric). Exceptions -> fail."""
    spec = TASKS.get(task_id)
    if not spec:
        return False, f"unknown task: {task_id}", None
    _kind, _weight, fn, _desc = spec
    try:
        return fn()
    except Exception as e:
        import traceback
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}", None
