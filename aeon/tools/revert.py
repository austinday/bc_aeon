"""Agent-callable rollback of a self-modification.

The restart path already auto-reverts a change that fails the pip/smoke/unit gates
DURING the transition. This tool covers the other case the gates can't catch: a
change that installs and passes tests but, once live, the agent observes to be a
behavioural regression (worse planning, a tool returning subtly wrong output).
``revert_aeon`` rolls the ``aeon/`` source back to a prior git checkpoint and then
restarts so the rolled-back code goes live — turning revert from
transition-only-automatic into an action the agent can take after live testing.
"""
import json
import os

from .base import BaseTool
from ..core import checkpoint

# Same staging file the restart machinery in main.py consumes.
RESTART_STATE_PATH = f"/tmp/aeon_restart_state_{os.getpid()}.json"


class RevertAeonTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name="revert_aeon",
            description=(
                "Roll the agent's OWN source code back to a previous checkpoint, then restart so "
                "the rolled-back code goes live. Use this when a self-modification passed verification "
                "but you observe, after restarting and live-testing, that it actually made the agent "
                "WORSE (a behavioural regression the test gates could not catch).\n"
                "Schema:\n"
                "  checkpoint (str, optional): The checkpoint tag to restore (see list_checkpoints / the "
                "**CHECKPOINTS** context). Defaults to the most recent checkpoint (undo the last change).\n"
                "  reason (str, optional): Why you are reverting.\n"
                'Example: {"tool_name": "revert_aeon", "parameters": {"reason": "new planner edit caused loops"}}'
            ),
        )
        self.worker = worker

    def _root(self):
        try:
            from ..core.paths import PROJECT_ROOT
            return str(PROJECT_ROOT)
        except Exception:
            return os.getcwd()

    def execute(self, checkpoint: str = None, reason: str = "Reverting a regressed self-modification") -> str:
        from ..core import checkpoint as ckpt  # local alias; param shadows the module name
        root = self._root()

        if not ckpt.is_git_repo(root):
            return ("Error: the Aeon source is not a git repository, so checkpoint-based revert is "
                    "unavailable. The restart machinery still auto-reverts changes that FAIL the "
                    "pip/smoke/unit gates; for a behavioural regression here, manually restore the "
                    "previous code with str_replace/write_file, then restart_aeon.")

        records = ckpt.list_checkpoints(root)
        if not records:
            return ("Error: no checkpoints exist yet. Checkpoints are created automatically at each "
                    "restart_aeon; there is nothing to roll back to.")

        target = checkpoint or records[0].get("tag")
        res = ckpt.restore_checkpoint(root, target)
        if not res.get("ok"):
            avail = ", ".join(r.get("tag", "?") for r in records[:8])
            return (f"Error: could not restore checkpoint '{target}': {res.get('reason')}. "
                    f"Available checkpoints: {avail}")

        # Stage a restart so the reverted source is reinstalled and relaunched. The
        # same path the restart_aeon tool uses; _execute_restart in main.py consumes it.
        try:
            state = self.worker.serialize_state()
            state["aeon_code_dir"] = os.path.abspath(root)
            state["original_cwd"] = os.getcwd()
            state["model_name"] = getattr(self.worker, "model_name", None)
            state["debug_mode"] = getattr(self.worker, "debug_mode", False)
            state["reason"] = f"REVERT to {target}: {reason}"
            with open(RESTART_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            return (f"Source was rolled back to '{target}', but staging the restart failed: {e}. "
                    f"Call restart_aeon manually to load the reverted code.")

        removed = res.get("deleted_added_files") or []
        extra = f" Removed {len(removed)} file(s) added since the checkpoint." if removed else ""
        return (f"Reverted the Aeon source to checkpoint '{target}'.{extra}\n"
                f"A restart is now staged: the agent will terminate and relaunch on the rolled-back "
                f"code (memories and history preserved). Reason: {reason}")
