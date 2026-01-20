from .base import BaseTool
from ..core.llm import LLMClient

# ANSI color codes
C_RESET = '\033[0m'
C_GREEN = '\033[92m'

class ThinkTool(BaseTool):
    """A tool for internal reasoning and planning."""
    def __init__(self, llm_client: LLMClient):
        super().__init__(
            name="think",
            description='Internal reasoning loop. Use to plan or analyze before acting. Params: `query` (str). Example: `{"tool_name": "think", "parameters": {"query": "Drafting plan to fix bug."}}`'
        )
        self.llm_client = llm_client

    def execute(self, query: str, working_memory: str):
        prompt = f"""Current Working Memory:
---
{working_memory}
---

Query to think about: \"{query}\"

Based on the working memory, provide a detailed thought process to address the query. This is for an internal monologue to help plan the next steps or understand a topic.
"""
        return self.llm_client.reason(prompt=prompt)


class SayToUserTool(BaseTool):
    """A tool to communicate with the user."""
    def __init__(self):
        super().__init__(
            name="say_to_user",
            description='Sends message to user. Use for updates, final answers, or questions. Params: `message` (str). Example: `{"tool_name": "say_to_user", "parameters": {"message": "Analysis complete."}}`'
        )

    def execute(self, message: str):
        # Print a newline to ensure the message starts below the tool call line
        print(f"\n{C_GREEN}{message}{C_RESET}")
        return f"Message delivered to user: {message}"