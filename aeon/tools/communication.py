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
            description=TOOL_DESC_THINK
        )
        self.llm_client = llm_client
        self.worker = worker

    def execute(self, query: str):
        working_memory = "No working memory available."
        if self.worker:
             working_memory = self.worker._format_open_files()
        
        prompt = THINK_TOOL_PROMPT.format(working_memory=working_memory, query=query)
        return self.llm_client.reason(prompt=prompt)


class SayToUserTool(BaseTool):
    """A tool to communicate with the user."""
    def __init__(self):
        super().__init__(
            name="say_to_user",
            description=TOOL_DESC_SAY_TO_USER
        )

    def execute(self, message: str):
        # Print a newline to ensure the message starts below the tool call line
        print(f"\n{C_GREEN}{message}{C_RESET}")
        return f"Message delivered to user: {message}"
