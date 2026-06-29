"""Git-backed checkpoints for safe, reversible self-modification.

The original restart path snapshots the ``aeon/`` package to a single PID-scoped
tarball in /tmp and deletes it on success — which protects exactly one restart
transition and nothing after it. This module replaces that with a durable,
multi-generation history using the git repo the source already lives in:

  * every self-modification can be checkpointed (a git tag + a jsonl index row),
  * checkpoints survive reboots and accumulate into a lineage you can diff,
  * recovery is a faithful restore of the ``aeon/`` subtree (modifications,
    deletions, AND additions) to a chosen checkpoint,
  * everything outside ``aeon/`` (the user's own project work) is never touched.

When the source dir is not a git repo, callers fall back to the legacy tarball
path; these functions all report that cleanly instead of raising.
"""
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Restore is deliberately scoped to the agent's own package so a rollback never
# clobbers unrelated user files that happen to live in the same repo.
_SCOPE = "aeon"
_TAG_PREFIX = "aeon-ckpt/"
_MAX_CHECKPOINTS = 30


def _git(root, *args, env=None):
    """Run a git command in ``root``; return (rc, stdout, stderr). Never raises."""
    try:
        run_env = None
        if env:
            run_env = dict(os.environ)
            run_env.update(env)
        p = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=60, env=run_env,
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "git executable not found"
    except subprocess.TimeoutExpired:
        return 124, "", "git command timed out"


def is_git_repo(root) -> bool:
    rc, out, _ = _git(root, "rev-parse", "--is-inside-work-tree")
    return rc == 0 and out == "true"


def _index_path(root) -> Path:
    return Path(root) / "aeon_output" / "checkpoints.jsonl"


def _snapshot_commit(root) -> str:
    """A commit SHA capturing the CURRENT working tree — tracked AND untracked
    (non-ignored) files — without touching the real index, worktree, or stash.

    ``git stash create`` is deliberately NOT used: it omits untracked files, so a
    brand-new file from a self-modification would be invisible to the checkpoint
    (and thus not removed on a later revert). Building the tree through a private
    temp index captures the full current state faithfully.
    """
    # GIT_INDEX_FILE must point at a path git can CREATE — a pre-existing empty file
    # is rejected ("index file smaller than expected"), so use a fresh temp dir and a
    # not-yet-existing index path inside it.
    tmpdir = tempfile.mkdtemp(prefix="aeon_ckpt_idx_")
    idx = os.path.join(tmpdir, "index")
    try:
        env = {"GIT_INDEX_FILE": idx}
        # Stage everything currently in the worktree (respects .gitignore).
        rc, _, err = _git(root, "add", "-A", env=env)
        if rc != 0:
            return ""
        rc, tree, _ = _git(root, "write-tree", env=env)
        if rc != 0 or not tree:
            return ""
        rc, head, _ = _git(root, "rev-parse", "HEAD")
        if rc == 0 and head:
            rc, sha, _ = _git(root, "commit-tree", tree, "-p", head, "-m", "aeon-checkpoint", env=env)
        else:  # repo with no commits yet
            rc, sha, _ = _git(root, "commit-tree", tree, "-m", "aeon-checkpoint", env=env)
        return sha if rc == 0 else ""
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def create_checkpoint(root, label: str = "self-mod") -> dict:
    """Tag the current working-tree state as a recoverable checkpoint.

    Returns a record dict with ``ok`` and (on success) ``tag``/``sha``. Best-effort:
    a non-git repo or any git failure returns ``ok=False`` with a reason, never raises.
    """
    root = Path(root)
    if not is_git_repo(root):
        return {"ok": False, "reason": "not a git repo"}
    sha = _snapshot_commit(root)
    if not sha:
        return {"ok": False, "reason": "could not capture working-tree state"}

    safe_label = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in label)[:40] or "self-mod"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{_TAG_PREFIX}{ts}-{safe_label}"
    rc, _, err = _git(root, "tag", "-f", tag, sha)
    if rc != 0:
        return {"ok": False, "reason": f"git tag failed: {err}"}

    rc, head, _ = _git(root, "rev-parse", "HEAD")
    record = {
        "tag": tag, "sha": sha, "label": label,
        "head": head if rc == 0 else "",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": time.time(),
    }
    try:
        idx = _index_path(root)
        idx.parent.mkdir(parents=True, exist_ok=True)
        with idx.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

    _prune(root)
    return {"ok": True, **record}


def list_checkpoints(root) -> list:
    """Checkpoints newest-first, filtered to tags that still exist in the repo."""
    root = Path(root)
    rc, out, _ = _git(root, "tag", "--list", f"{_TAG_PREFIX}*")
    live = set(out.splitlines()) if rc == 0 else set()
    records, seen = [], set()
    idx = _index_path(root)
    try:
        if idx.exists():
            for line in idx.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                tag = r.get("tag")
                if tag and tag in live and tag not in seen:
                    records.append(r)
                    seen.add(tag)
    except Exception:
        pass
    # Include any live tags missing from the index (e.g. created out of band).
    for tag in live:
        if tag not in seen:
            records.append({"tag": tag, "sha": "", "label": "", "created_at": "", "epoch": 0})
    records.sort(key=lambda r: r.get("epoch", 0), reverse=True)
    return records


def _ls_tree(root, sha: str, scope: str) -> list:
    """Files under ``scope`` present in commit ``sha``."""
    rc, out, _ = _git(root, "ls-tree", "-r", "--name-only", sha, "--", scope)
    return [l for l in out.splitlines() if l.strip()] if rc == 0 else []


def _ls_worktree(root, scope: str) -> list:
    """Files under ``scope`` present in the working tree now (tracked + untracked,
    excluding .gitignored)."""
    files = []
    rc, tracked, _ = _git(root, "ls-files", "--", scope)
    if rc == 0:
        files += [l for l in tracked.splitlines() if l.strip()]
    rc, untracked, _ = _git(root, "ls-files", "--others", "--exclude-standard", "--", scope)
    if rc == 0:
        files += [l for l in untracked.splitlines() if l.strip()]
    return files


def _resolve(root, ref: str) -> str:
    """Resolve a checkpoint tag (or raw SHA) to a commit SHA, or '' if unknown."""
    if not ref:
        return ""
    rc, sha, _ = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    return sha if rc == 0 else ""


def restore_checkpoint(root, ref: str) -> dict:
    """Faithfully restore the ``aeon/`` subtree to checkpoint ``ref``.

    Handles modifications and deletions (via ``git checkout``) AND additions
    (files created since the checkpoint are removed), so the package matches the
    checkpoint exactly. Files outside ``aeon/`` are left untouched.
    """
    root = Path(root)
    if not is_git_repo(root):
        return {"ok": False, "reason": "not a git repo"}
    sha = _resolve(root, ref)
    if not sha:
        return {"ok": False, "reason": f"unknown checkpoint: {ref}"}

    # Everything currently under aeon/ (tracked + untracked, ignoring .gitignored),
    # vs. everything the checkpoint had under aeon/. Files present now but absent in
    # the checkpoint — whether they were tracked-added or just dropped in untracked —
    # must be removed so the restored tree matches the checkpoint exactly.
    snap = set(_ls_tree(root, sha, _SCOPE))
    current = set(_ls_worktree(root, _SCOPE))

    rc, _, err = _git(root, "checkout", sha, "--", _SCOPE)
    if rc != 0:
        return {"ok": False, "reason": f"git checkout failed: {err}"}

    removed = []
    for rel in sorted(current - snap):
        try:
            (root / rel).unlink()
            removed.append(rel)
        except OSError:
            pass
    # Drop the just-restored paths from the index so `git status` reflects reality.
    _git(root, "reset", "--quiet", "--", _SCOPE)
    return {"ok": True, "sha": sha, "ref": ref, "deleted_added_files": removed}


def diff_checkpoint(root, ref: str, max_chars: int = 4000) -> str:
    """A bounded ``git diff`` of the current ``aeon/`` tree vs checkpoint ``ref``."""
    root = Path(root)
    sha = _resolve(root, ref)
    if not sha:
        return f"(unknown checkpoint: {ref})"
    rc, out, _ = _git(root, "--no-pager", "diff", sha, "--", _SCOPE)
    if rc != 0:
        return "(diff unavailable)"
    if not out:
        return "(no differences from this checkpoint)"
    return out if len(out) <= max_chars else out[:max_chars] + "\n... [diff truncated] ..."


def _prune(root):
    """Keep only the newest ``_MAX_CHECKPOINTS`` tags so the lineage stays bounded."""
    records = list_checkpoints(root)
    for r in records[_MAX_CHECKPOINTS:]:
        tag = r.get("tag")
        if tag:
            _git(root, "tag", "-d", tag)
