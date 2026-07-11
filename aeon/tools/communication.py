from .base import BaseTool
from ..core.llm import LLMClient
from ..core.prompts import (
    TOOL_DESC_THINK,
    TOOL_DESC_SAY_TO_USER,
    THINK_TOOL_PROMPT,
)

# ANSI color codes
C_RESET = '\033[0m'
C_GREEN = '\033[92m'

class ThinkTool(BaseTool):
    """A tool for internal reasoning and planning."""
    def __init__(self, llm_client: LLMClient, worker=None):
        super().__init__(
            name="think",
            description=TOOL_DESC_THINK,
            underlying_model=llm_client.model if llm_client else None
        )
        self.llm_client = llm_client
        self.worker = worker

    # Per-file cap for the working-memory snapshot fed to think. Without a bound,
    # several large open files could blow the context window and slow the call;
    # the agent already has the files in its main context, so think only needs a
    # generous-but-bounded view.
    THINK_FILE_CAP = 60000

    def execute(self, query: str):
        if not query or not str(query).strip():
            return "Error: 'query' parameter is required (what should I think about?)."
        working_memory = "No working memory available."
        if self.worker:
            working_memory = self.worker._format_open_files(max_content_len=self.THINK_FILE_CAP)

        prompt = THINK_TOOL_PROMPT.format(working_memory=working_memory, query=query)
        return self.llm_client.reason(prompt=prompt)


class SayToUserTool(BaseTool):
    """A tool to communicate with the user."""
    def __init__(self, worker=None):
        super().__init__(
            name="say_to_user",
            description=TOOL_DESC_SAY_TO_USER
        )
        self.worker = worker

    def execute(self, message: str):
        if not message:
            return "Error: message parameter is required."
        # Print a newline to ensure the message starts below the tool call line
        print(f"\n{C_GREEN}{message}{C_RESET}")
        # Stash the delivered text on the worker: for a sub-agent, the final
        # say_to_user IS the report the principal reads (the wrapper prefers this
        # over last_observation, which by the terminal turn only holds the
        # PREVIOUS turn's output — the deliverable used to be lost entirely).
        if self.worker is not None:
            self.worker.last_say_to_user = message
        # Return only a concise confirmation, NOT the full message: the message
        # text is already in this action's parameters, so echoing it back would
        # duplicate a potentially long report into the agent's context every turn.
        chars = len(message)
        return f"Message delivered to user ({chars:,} chars)."
