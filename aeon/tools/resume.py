from .base import BaseTool


class ResumePreviousSessionTool(BaseTool):
    """Resume a previous, interrupted session's objective from its saved state dump.

    The state dump is written when a run is stopped (e.g. Ctrl+C). This tool loads
    it back — objective, plan, memories, and attempt history — so the agent
    continues that work instead of starting over. It is invoked when the USER asks
    to continue/resume (the agent recognizes phrasings like 'continue from where
    you left off' and calls this on turn 1)."""

    def __init__(self, worker):
        super().__init__(
            name="resume_previous_session",
            description=(
                "Resume the PREVIOUS session that was stopped/interrupted in this workspace, picking up "
                "its objective, plan, memories, and attempt history exactly where it left off. Call this "
                "ONLY when the user asks you to continue or resume prior work — e.g. 'continue from where "
                "you left off', 'resume the previous task', 'pick up where you stopped', 'keep going on "
                "what you were doing'. It reads the state dump saved when the last run was stopped (such "
                "as by Ctrl+C) and restores it. The user's current message is integrated with the "
                "restored objective — a plain 'continue' resumes the same goal, while a request that "
                "redirects or extends it (e.g. 'continue but now also do X') reshapes the objective "
                "accordingly — so you continue that work instead of treating the message as a brand-new "
                "task. If nothing was saved, it says so.\n"
                "Schema: no parameters.\n"
                "Example: {\"tool_name\": \"resume_previous_session\", \"parameters\": {}}"
            ),
        )
        self.worker = worker

    def execute(self) -> str:
        if not self.worker:
            return "Error: Worker context missing."
        return self.worker.resume_from_dump()
